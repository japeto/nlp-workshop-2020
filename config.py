import datetime
"""Config file

Setting source vectors and output folder
"""

# File with word embeddings. Binary format from word2vec and gensim libraries is used.
word2vec_file = '/Users/macbookpro/Downloads/GoogleNews-vectors-negative300.bin'

# File where results of our method are being written by default.
date_object = str(datetime.date.today())
output_file = f'outputs/output{date_object}.txt'
