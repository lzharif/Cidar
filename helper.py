import glob

class Helper:
    def listFiles(self, folder):
        types = ('/*.jpg', '/*.png')
        files = []
        for tipe in types:
            a = folder + tipe
            files.extend(glob.glob(folder + tipe))
        return files