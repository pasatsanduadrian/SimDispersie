# Flask Dispersion Map Demo

Vizualizează dispersia poluanților pe hartă reală OSM, cu suprapunere color și simulare meteorologică pentru 24h, totul direct din browser!
Acest proiect este licențiat sub licența MIT. Vezi fișierul LICENSE pentru detalii.


---

## 🛠️ Instalare și rulare locală

**Cerințe:**
- Python 3.8+ (recomandat 3.10+)
- `pip install -r requirements.txt` (vezi mai jos pachetul de bază)

**Pași:**

1. Clonează repo sau descarcă fișierele:
    ```sh
    git clone <adresa_repo>
    cd <adresa_repo>
    ```

2. Instalează dependențele:
    ```sh
    pip install flask osmnx contextily pyngrok tqdm python-dotenv matplotlib numpy==1.24.4 pandas
    ```

3. Configurează `.env`:
    - Copiază `.env.example` în `.env` și completează tokenul ngrok dacă vrei acces public.
    - Poți rula și doar local fără ngrok.

4. Pornește aplicația:
    ```sh
    python app.py
    ```
    - Vei vedea link-ul local sau public (dacă ai completat NGROK_TOKEN).

---

## 🟢 Utilizare rapidă Google Colab

- Încarcă fișierul `app.py` într-un notebook Colab.
- Instalează dependențele:
    ```python
    !pip install flask osmnx contextily pyngrok tqdm python-dotenv matplotlib numpy==1.24.4 pandas
    ```
- (Opțional, dacă vrei acces public prin ngrok) Creează `.env` cu tokenul tău:
    ```python
    with open('.env', 'w') as f:
        f.write('NGROK_TOKEN=tokenul_tău_aici\n')
        f.write("NGROK_HOSTNAME=stable-guided-buck.ngrok-free.app\n")
    ```
    sau direct:
    ```python
    import os
    os.environ['NGROK_TOKEN'] = 'tokenul_tău_aici'
    ```
- Rulează aplicația:
    ```python
    !python app.py
    ```
- Accesează link-ul din output (ngrok sau local) pentru a folosi aplicația din browser.


## 🔑 Variabile de mediu

- `.env.example` este modelul de fișier
- Variabila `PORT` permite schimbarea portului (implicit 5015).
- Dacă nu folosești ngrok, aplicația rulează doar local.

---

## 📄 Structura fișierelor

- `app.py` – codul aplicației Flask, totul într-un singur fișier.
- `.env.example` – model de configurare variabile de mediu.
- Variabila `PORT` permite schimbarea portului (implicit 5015).
- `.gitignore` – exclude .env și fișiere temporare.
- `README.md` – acest fișier.

---

## ✨ Demo vizual

![Poza mea](https://drive.google.com/uc?export=view&id=14KNkkpfzyX6dGSO_J9BawYZldjVsAeTm)

---

## ❗ Note și troubleshooting

- Dacă basemap-ul apare **gri/monocrom**, asigură-te că ai conectivitate la internet (contextily descarcă tile-urile OSM).
- Pentru acces din rețea/public, folosește ngrok și nu uita să pui tokenul în `.env`.
- Pentru rulare pe server, folosește procese gen `gunicorn`/`waitress` și reverse proxy (avansat).
- Dacă întâmpini erori de tip *"numpy.dtype size changed"*, reinstalează
  pachetele `numpy` și `pandas` cu versiunile din `requirements.txt` (de ex.
  `pip install numpy==1.24.4 pandas==1.5.3`).

### 🛠️ Notă utilă pentru depanare (Colab/ngrok)
Dacă primești o eroare de tip **„port ocupat”** sau tunelul ngrok nu mai pornește corect (de exemplu după reporniri repetate în Google Colab):

```python
!fuser -k 5015/tcp
from pyngrok import ngrok
ngrok.kill()
```
Aceasta va elibera portul si va inchide tunelul ngrok.

## 🔬 Teste
Rulati `pytest` pentru a executa testele unitare.

## 🐳 Docker
Rulati `docker build -t dispersie .` apoi `docker run -p 5015:5015 dispersie`.
