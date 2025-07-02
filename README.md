# bike-sharing-api

## Descrizione
API per la gestione e l'analisi di un servizio di bike sharing. Permette di caricare dataset, addestrare un modello di machine learning, ottenere statistiche e fare predizioni sul numero di utenti.

## Requisiti

- **Docker** e **Docker Compose**

## Installazione e Avvio (con Docker)

1. **Clona il repository**
   ```bash
   git clone https://github.com/danielecanzoneri/bike-sharing-api
   cd bike-sharing-api
   ```

2. **Crea un file `.env` nella root del progetto** con il seguente contenuto (puoi personalizzare le variabili):
   ```
   POSTGRES_USER=youruser
   POSTGRES_PASSWORD=yourpassword
   POSTGRES_DB=bikesharing
   ```

3. **Avvia i servizi**
   ```bash
   docker compose up --build
   ```
   L'API sarà disponibile su [http://localhost:8000](http://localhost:8000).

## Utilizzo delle API

- **/load** (POST): carica un dataset CSV nel database.
- **/train** (GET): addestra il modello sui dati caricati.
- **/stats/avg** (GET): statistiche medie per ora/giorno.
- **/stats/total** (GET): totale utenti per mese.
- **/predict** (POST): predice il numero di utenti su nuovi dati CSV.
- **/predict/{prediction_id}** (GET): scarica le predizioni in formato CSV.

Per dettagli e test degli endpoint, visita la documentazione interattiva su [http://localhost:8000/docs](http://localhost:8000/docs).