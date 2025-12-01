import xlwings as xw

def main():
wb = xw.Book.caller() # Permet d'accéder au classeur Excel appelant
sheet = wb.sheets[0]
valeur = sheet.range("A1").value
sheet.range("B1").value = valeur * 2

if __name__ == "__main__":
xw.Book("ton_fichier.xlsx").set_mock_caller()
main()
