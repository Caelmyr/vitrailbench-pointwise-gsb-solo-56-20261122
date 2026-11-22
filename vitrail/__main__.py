import os

from waitress import serve

from .http import create_app


def main():
    host=os.environ.get("HOST","0.0.0.0")
    port=int(os.environ.get("PORT","8000"))
    serve(create_app(),host=host,port=port,threads=5,channel_timeout=120,max_request_body_size=54_000_000)


if __name__=="__main__":
    main()
