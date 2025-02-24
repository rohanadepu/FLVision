#########################################################
#    Imports / Env setup                                #
#########################################################

import flwr as fl
import argparse


def main():

    #########################################################
    #    Config Server                                      #
    #########################################################

    parser = argparse.ArgumentParser(
        description="Federated Learning Training Script Server Side, --rounds [1-10] to select rounds")
    parser.add_argument("--rounds", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=1,
                        help="Rounds of training 1-10")

    args = parser.parse_args()

    roundInput = args.rounds

    #########################################################
    #    Initialize Server                                  #
    #########################################################

    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=roundInput)
    )


# execute
if __name__ == "__main__":
    main()
