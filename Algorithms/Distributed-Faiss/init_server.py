from config import INDEX_ALREADY_CREATED, PATH_DATASETS
from distributed_faiss.server import IndexServer

SERVER_ID = 0
SERVER_PORT = 8081

def main():
    server = IndexServer(SERVER_ID, PATH_DATASETS)
    server.start_blocking(SERVER_PORT, load_index=INDEX_ALREADY_CREATED)

if __name__ == "__main__":
    main()