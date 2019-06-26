import requests
import shutil


def download_file(download_URL, filename):
    """Download file from CURL url using request.
    download_URL: """
    with requests.get(download_URL, stream=True) as r:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-du",      "--download_URL",   help="The URL to download the file.", default='https://api.figshare.com/v2/file/download/15593951')
    parser.add_argument("-of",      "--output_file",      help="The number of workers to use when running the analysis.", default='data/commits_by_org.feather')
    args = parser.parse_args()

    download_file(args.download_URL, args.output_file)
