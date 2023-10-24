from config import INDEX_ALREADY_CREATED, PATH_INDEX
from distributed_faiss.server import IndexServer

''' ### SINGLE SERVER CONFIG ### '''
SERVER_ID = 0       # different for each server
SERVER_PORT = 8081

def main():
    server = IndexServer(SERVER_ID, PATH_INDEX)
    server.start_blocking(SERVER_PORT, load_index=INDEX_ALREADY_CREATED)

if __name__ == "__main__":
    main()