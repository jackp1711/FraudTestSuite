import os


class File_writer:

    file_name = ""

    def __init__(self):
        self.file_name = "results2.txt"

        if not os.path.isfile(self.file_name):
            f = open(self.file_name, 'w')
            f.write("Results" + '\n')
            f.close()

    def write_file(self, text):
        with open(self.file_name, 'a', encoding='utf-8') as file:
            file.write(text + '\n')
