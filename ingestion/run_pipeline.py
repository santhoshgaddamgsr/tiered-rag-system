from download_arxiv import download_papers
from ingest import main

download_papers(
    query="large language models",
    start_year=2023,
    end_year=2023,
    max_results=100
)

main()