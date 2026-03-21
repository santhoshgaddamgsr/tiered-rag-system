import arxiv
import os

DOWNLOAD_PATH = "data/raw_docs"


def download_papers(query, start_year, end_year, max_results=100):

    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    # arXiv date range query
    date_query = f"submittedDate:[{start_year}0101 TO {end_year}1231]"

    full_query = f"{query} AND {date_query}"

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    downloaded = 0
    skipped = 0

    for result in search.results():

        filename = result.title.replace(" ", "_").replace("/", "") + ".pdf"
        path = os.path.join(DOWNLOAD_PATH, filename)

        # Skip if already downloaded
        if os.path.exists(path):
            print(f"Skipping (already exists): {filename}")
            skipped += 1
            continue

        print(f"Downloading: {result.title} ({result.published.year})")

        result.download_pdf(
            dirpath=DOWNLOAD_PATH,
            filename=filename
        )

        downloaded += 1

    print("\nDownload completed.")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")