#########################################################
#    Imports / Env setup                                #
#########################################################

import flwr as fl
import argparse


#########################################################
#    Config Server                                      #
#########################################################

def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning Training Script Server Side, --rounds [1-10] to select rounds")
    parser.add_argument("--rounds", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=8,
                        help="Rounds of training 1-10")
    parser.add_argument("--min_clients", type=int, choices=[1, 2, 3, 4, 5, 6], default=2,
                        help="Minimum number of clients required for training")

    args = parser.parse_args()

    roundInput = args.rounds
    minClients = args.min_clients

    #########################################################
    #    Initialize Server                                  #
    #########################################################

    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=roundInput),
        strategy=fl.server.strategy.FedAvg(
            min_fit_clients=minClients,
            min_evaluate_clients=minClients,
            min_available_clients=minClients
        )
    )


if __name__ == "__main__":
    main()
