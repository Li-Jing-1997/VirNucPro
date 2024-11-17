import requests
from bs4 import BeautifulSoup
import random
import os
from tqdm import tqdm

urls = ["https://ftp.ncbi.nlm.nih.gov/refseq/release/bacteria/", "https://ftp.ncbi.nlm.nih.gov/refseq/release/archaea/",
        "https://ftp.ncbi.nlm.nih.gov/refseq/release/fungi/", "https://ftp.ncbi.nlm.nih.gov/refseq/release/protozoa/",
        "https://ftp.ncbi.nlm.nih.gov/refseq/release/plant/", "https://ftp.ncbi.nlm.nih.gov/refseq/release/invertebrate/",
        "https://ftp.ncbi.nlm.nih.gov/refseq/release/vertebrate_mammalian/", "https://ftp.ncbi.nlm.nih.gov/refseq/release/vertebrate_other/"
        ]

need_more_data_list = ['plant', 'invertebrate', 'vertebrate_mammalian', 'vertebrate_other']

random.seed(42)
files_to_download = set()

for utl_type in urls:
    file_urls = []
    if any(keyword in utl_type for keyword in need_more_data_list):
        response = requests.get(utl_type)
        soup = BeautifulSoup(response.content, "html.parser")
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            full_link = utl_type + link if not link.startswith("http") else link
            if 'genomic' in full_link:
                file_urls.append(full_link)
        gbff_files = [utl_type for utl_type in file_urls if utl_type.endswith("genomic.gbff.gz")]
        sampled_gbff_files = random.sample(gbff_files, min(10, len(gbff_files)))
        print(sampled_gbff_files)

        for gbff_file in sampled_gbff_files:
            files_to_download.add(gbff_file)

            files_to_download.add(gbff_file.replace("genomic.gbff.gz", "1.genomic.fna.gz"))
    else:
        response = requests.get(utl_type)
        soup = BeautifulSoup(response.content, "html.parser")
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            full_link = utl_type + link if not link.startswith("http") else link
            if 'genomic' in full_link:
                file_urls.append(full_link)
        gbff_files = [utl_type for utl_type in file_urls if utl_type.endswith("genomic.gbff.gz")]
        sampled_gbff_files = random.sample(gbff_files, min(1, len(gbff_files)))
        print(sampled_gbff_files)

        for gbff_file in sampled_gbff_files:
            files_to_download.add(gbff_file)

            files_to_download.add(gbff_file.replace("genomic.gbff.gz", "1.genomic.fna.gz"))

def download_file(url):
    os.makedirs('data/', exist_ok=True)
    local_filename = 'data/' + url.split('/')[-1]

    response = requests.head(url)
    remote_size = int(response.headers.get('content-length', 0))

    if os.path.exists(local_filename):
        local_size = os.path.getsize(local_filename)

        if local_size == remote_size:
            print(f"{local_filename} has been downloaded and the size matches, skipping...")
            return
        else:
            print(f"{local_filename} exists but the size does not match. Re-downloading...")
            os.remove(local_filename)

    print(f"Downloading: {local_filename}")
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 8192  # Chunk size
        with open(local_filename, 'wb') as f, tqdm(
            desc=local_filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=block_size):
                f.write(chunk)
                bar.update(len(chunk))
    
    print(f"{local_filename} finished")

download_file('https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.1.1.genomic.fna.gz')
download_file('https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.1.genomic.gbff.gz')

for file_url in files_to_download:
    download_file(file_url)