import argparse
import flwr as fl

parser = argparse.ArgumentParser(description='Select to enable secure aggregation')
parser.add_argument('--secAggP', action='store_true', help='Enable model adversarial training')

args = parser.parse_args()

secAggPlusEnabled = args.secAggP

if secAggPlusEnabled:
    print("Secure Aggregation Plus Enabled \n")
else:
    print("Secure Aggregation Plus Disabled \n")

def main():
    if secAggPlusEnabled:
        secagg_config = fl.common.SecAggConfig(
            secagg_type="SecAgg+",
            threshold_ratio=0.5,
        )
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(
                num_rounds=10,
                secagg_config=secagg_config
            )
        )
    else:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(
                num_rounds=10
            )
        )

if __name__ == "__main__":
    main()
