from crawl4ai import *
from crawl4ai import CrawlerMonitor, DisplayMode
from typing import List, Dict, Any
from dataclasses import dataclass
import asyncio
import datetime 
import json 
import os
from dotenv import load_dotenv
from urllib.parse import urlparse 
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

visited = dict()
@dataclass
class ProcessedChunk:
	url : str
	chunk_number: int
	title: str
	summary: str
	content: str
	metadata: dict
	embedding: List[float]

#print(os.getenv("OPENAI_API_KEY"), os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SECRET_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SECRET_KEY"))


async def crawl4urls():
	run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
		exclude_external_links=True,
        stream=True  # Default: get all results at once
    )

	dispatcher = MemoryAdaptiveDispatcher(memory_threshold_percent=90,
									   max_session_permit=40, 
									   monitor = CrawlerMonitor(display_mode=DisplayMode.AGGREGATED, max_visible_rows=15,))	
	
	urls = ["https://docs.pwntools.com/en/stable/"]
	f = open("urls.txt", "w")
	f.write(urls[0] + "\n")
	async with AsyncWebCrawler() as crawler:
		while len(urls):
			async for result in await crawler.arun_many(
				urls=urls,
				config=run_config,
				dispatcher=dispatcher
			):
				for url in urls:
					print(url)
					visited[url] = True
				urls = []
				for l in result.links["internal"]:
					link = l["href"]
					if "#" in link: 
						continue
					if visited.get(link) is None:
						urls.append(link)
						f.write(link + "\n")
						visited[link] = True
	f.close()
			
def split_into_chunks(s : str, chunk_size : int):
	chunks = []
	start = 0
	text_length = len(s)
	while start < text_length:
		end = start + chunk_size
		if end >= text_length:
			chunks.append(s[start:])
			break
		
		# try to find a code block first
		chunk = s[start:end]
		code_block = chunk.rfind("```")
		if code_block != -1 and code_block > chunk_size * 0.3: # if the code block is too small, we probably don't want it to be a separate chunk by itself
			end = start + code_block
		elif "\n\n" in chunk: # paragraph break
			last_break = chunk.rfind("\n\n")
			if last_break > chunk_size * 0.3:
				end = start + last_break
		elif ". " in chunk: # don't break in the middle of a sentence
			last_break = chunk.rfind(". ")  
			if last_break > chunk_size * 0.3:
				end = start + last_break + 1
		chunk = s[start:end].strip()
		chunks.append(chunk)
		start = max(start + 1, end)

	return chunks

async def get_title_and_summary(chunk : str, url : str):
	"""Extract title and summary using GPT-4."""
	system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
	try:
		response = await openai_client.chat.completions.create(
			model = os.getenv("LLM_MODEL", "gpt-4o-mini"),
			messages = [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": f"URL: {url}\n\nContent: {chunk[:1000]}..."}
			],
			response_format = {"type": "json_object"}
		)
		return json.loads(response.choices[0].message.content)
	except Exception as e:
		print(f"Error getting title and summary: {e}")
		return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(chunk : str):
	"""Get the embedding of a chunk using GPT-4."""
	try:
		response = await openai_client.embeddings.create(
			model = "text-embedding-3-small",
			input = chunk
		)
		return response.data[0].embedding
	except Exception as e:
		print(f"Error getting embedding: {e}")
		return [0.0] * 1536


async def process_chunk(chunk : str, i : int, url : str):
	"""
		Process a chunk of text and store it into the ProcessedChunk class defined above
	"""
	extracted = await get_title_and_summary(chunk, url)
	embedding = await get_embedding(chunk)

	metadata = {
		"chunk_size": len(chunk),
		"url": urlparse(url).path,
		"source": "pwntools",
		"crawled_at": datetime.datetime.now().isoformat(),
	}

	return ProcessedChunk(
		url = url,
		chunk_number = i, 
		title = extracted["title"], 
		summary = extracted["summary"], 
		content = chunk, 
		metadata = metadata, 
		embedding = embedding
	)

async def store_chunk(chunk : ProcessedChunk):
	"""Insert a procssed chunk into Supabase"""
	try:
		data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
		# TODO: insert into supabase 
		result = supabase.table("site_pages").insert(data).execute()
		print(f"Stored chunk: {chunk.chunk_number} for {chunk.url}")
		return result
	
	except Exception as e:
		print(f"Error storing chunk: {e}")
		return None

async def process_url(result : str, url : str):
	chunks = split_into_chunks(result, 5000)	
	#print(chunks)
	chunks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
	processed_chunks = await asyncio.gather(*chunks)
	#print(processed_chunks)

	stored_chunks = [store_chunk(chunk) for chunk in processed_chunks]
	await asyncio.gather(*stored_chunks)


					
async def process_urls():
	with open("urls.txt", "r") as f:
		urls = f.readlines()

	browser_config = BrowserConfig(headless=True, verbose=False)
	run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
		verbose=True,
        stream=True  # process each url as soon as it is crawled
    )
	dispatcher = MemoryAdaptiveDispatcher(memory_threshold_percent=90, 
									  max_session_permit=40, 
									  monitor=CrawlerMonitor(display_mode=DisplayMode.AGGREGATED, max_visible_rows=15,))

	async with AsyncWebCrawler(config=browser_config) as crawler:
		async for result in await crawler.arun_many(
			urls=urls,
			config=run_config,
			dispatcher=dispatcher
		):
			if result.success:
				#print("Processing: " + result.markdown_v2.raw_markdown)
				await process_url(result.markdown_v2.raw_markdown, result.url)
				#print("Processed: " + result.url)
			else:
				print(f"Failed to crawl {result.url}: {result.error_message}")




if __name__ == "__main__":
    #asyncio.run(crawl4urls())
	asyncio.run(process_urls())
