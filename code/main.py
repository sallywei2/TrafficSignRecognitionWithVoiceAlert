from CustomArgParser import CustomArgParser

def main(*args):
    parser = CustomArgParser()
    args = parser.parse_args()
    vargs = vars(args)

    if vargs['train'] == True:
        print("TODO: Download training data. train model, and save weights")
    elif vargs['results'] == True:
        print("TODO: Plot/show the results of the model after testing using the saved weights")

if __name__ == '__main__':
    main()