# Detekcja gestów
## Autorzy
Kamil Śliwa, Kamil Węgrzyn, Marcin Wiśniowski, Sławomir Zdunek

## Opis
Projekt rozpoznający wybrane gesty w czasie rzeczywistym z zastosowaniem klasycznych metod wizji komputerowej takich jak: zamknięcie, otwarcie, Convex Hull, Convexity Defects i tym podobne.

Program uczy się na zdjęciach wykonanych na jednolitym tle, które są przycięte do nadgarstka.

## Instrukcja instalacji i uruchomienia
Należy zainstalować Python na komputer (tutorial, jak to zrobić): https://wiki.python.org/moin/BeginnersGuide(2f)Download.html

Następnie pobrać pliki z tego repozytorium do folderu.

W folderze otworzyć konsolę i wpisać komendy:
`pip install opencv-python`,
`pip install numpy`,
`pip install matplotlib`.

Jeżeli komendy nie działają, najprawdopodobniej należy zainstalować pakiet instalacyjny pip: https://pip.pypa.io/en/stable/installation/

Na końcu można odpalić już projekt wpisując w konsolę: `py main.py`.
