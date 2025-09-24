# process_file.py
import sys

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Example: Process the file (here we reverse the lines)
            outfile.write(line[::-1])

if __name__ == '__main__':
    main()

