# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from os import path
import argparse
import requests
import shutil

# Parse command line arguments
parser = argparse.ArgumentParser(description="Download and store images from URLs in 'textfile'.")
parser.add_argument('textfile', help='Plaintext file containing URLs.', type=str)
args = parser.parse_args()
file = args.textfile

# context manager: read text file
with open(file, mode='r') as fileobj:
    # loop over lines
    for line in fileobj:
        # replace escape character
        line = line.rstrip('\n')

        # check for empty string
        if line:
            # Infer filename from url
            filename = path.basename(line)

            # Create response object (better use context manager to handle requests)
            r = requests.get(line, stream=True)

            # check for succesful request
            if r.status_code == 200:
                # context manager: open file object to write stream into it
                with open(filename, mode='wb') as file:
                    # force decompression
                    r.raw.decode_content = True

                    # stream url into file object
                    shutil.copyfileobj(r.raw, file)
            r.close()
