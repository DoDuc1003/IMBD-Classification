import torchtext.dataset

def get_IMDB_dataset():
    train, test = torchtext.dataset.IMDB.split()
    
    return train, test