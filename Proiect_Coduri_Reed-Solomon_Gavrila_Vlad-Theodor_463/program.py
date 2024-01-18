import reedsolo

# Setăm parametrii pentru codul Reed-Solomon
n = 255  # Lungimea totală a blocului codificat (inclusiv simbolurile de paritate)
k = 223  # Lungimea blocului de date

# Creăm instanța de codificator/decodificator Reed-Solomon
rs = reedsolo.RSCodec(n-k)

# Exemplu de mesaj pentru codificare
mesaj = b"Exemplu de mesaj pentru codificarea Reed-Solomon RS(255, 223)"
mesaj_padded = mesaj.ljust(k, b'\0')  # Umplerea mesajului până la lungimea k, dacă este necesar

# Codificarea mesajului
mesaj_codificat = rs.encode(mesaj_padded)

# Simulăm o eroare în mesajul codificat
mesaj_erorat = bytearray(mesaj_codificat)
mesaj_erorat[10] = 0xFF  # Introducem o eroare

# Decodificăm mesajul și corectăm eroarea
mesaj_decorectat = rs.decode(mesaj_erorat)

print("Mesajul original: ", mesaj)
print("Mesajul decodificat: ", mesaj_decorectat)