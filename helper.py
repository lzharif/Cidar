import glob

# Kelas penolong
class Helper:

	# Fungsi untuk melihat isi dari suatu folder
    def listFiles(self, folder):
        types = ('/*.jpg', '/*.png')
        files = []
        for tipe in types:
            a = folder + tipe
            files.extend(glob.glob(folder + tipe))
        return files