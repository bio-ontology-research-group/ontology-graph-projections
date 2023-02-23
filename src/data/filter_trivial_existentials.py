import sys

bot = "http://www.w3.org/2002/07/owl#Nothing"
top = "http://www.w3.org/2002/07/owl#Thing"
def main(input_file):
    non_trivial = list()
    with open(input_file, "r") as f:
        for line in f.readlines():
            stripped = line.rstrip("\n")
            if stripped.startswith(bot):
                continue
            if stripped.endswith(top):
                continue
                            
            non_trivial.append(line)

    output_file = input_file.replace(".csv", "_non_trivial.csv")
    with open(output_file, "w") as f:
        for line in non_trivial:
            f.write(line)

if __name__ == "__main__":
    input_file = sys.argv[1]
    if not input_file.endswith(".csv"):
        raise ValueError("Input file must be a CSV file")
    main(input_file)
