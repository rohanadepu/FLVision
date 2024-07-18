import argparse
import flwr as fl


def main():
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=10
        )
    )


if __name__ == "__main__":
    main()
