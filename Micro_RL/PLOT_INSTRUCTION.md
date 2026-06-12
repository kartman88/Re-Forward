# Regole Principali per Creare Figure Scientifiche di Alta Qualità

## 1. Formato, Dimensione e Risoluzione

* 
**Controllare dimensioni e proporzioni:** Evitare che le dimensioni dei grafici siano arbitrarie o dipendano dai dati; impostare una dimensione fissa e un rapporto d'aspetto bloccato (es. una tela quadrata di 1000x1000 pixel) per mantenere uniformità tra tutte le figure del paper.


* 
**Risoluzione manuale:** Le risoluzioni di default dei linguaggi di programmazione (es. 640x480 pixel in Python) sono troppo basse e portano a immagini sgranate.


* 
**Grafica vettoriale:** Salvare sempre i grafici finali in formati vettoriali scalabili (PDF o SVG) e mai in formati raster (come PNG), per garantire una qualità infinita e l'assenza di pixel.



## 2. Gestione dei Limiti di Plotting (Zoom)

* 
**Evitare i range automatici:** Le librerie grafiche tendono a inquadrare tutti gli elementi presenti (incluse linee di regressione molto lunghe o spazi bianchi), allontanando la visuale dai dati reali.


* 
**Calcolare il centro:** Trovare manualmente il punto medio (mediana) e il range delle componenti orizzontali e verticali dei dati principali.


* 
**Creare un riquadro di zoom:** Impostare i nuovi limiti degli assi (X e Y) partendo dal punto medio e aggiungendo un parametro di "zoom" manuale, creando così un'inquadratura perfettamente centrata sui dati più importanti e tagliando via le code inutili.



## 3. Tipografia (Font e Testi)

* 
**Font scientifici:** Evitare i font predefiniti (come Calibri o DejaVu Sans), prediligendo font più formali e professionali come Courier New o Times New Roman.


* 
**Dimensioni proporzionate:** Impostare dimensioni del testo apparentemente grandi in fase di codice (es. 20 punti).


* 
**Test di riduzione:** Tenere conto che in un tipico articolo a due colonne la figura verrà rimpicciolita, e la dimensione finale del testo sul documento non dovrebbe mai scendere sotto gli 8 punti per rimanere leggibile.



## 4. Griglie e Linee di Riferimento

* 
**Supporto all'analisi:** L'uso di griglie maggiori e minori può aiutare i lettori ad analizzare porzioni specifiche di dati e associare i valori esatti agli assi.


* 
**Leggerezza e trasparenza:** Per mantenere un aspetto professionale senza che la griglia sovrasti i dati, impostare linee molto sottili (es. spessore da 0.25 a 0.75) e usare un'alta trasparenza (parametro "alpha") in modo che risultino di un grigio molto chiaro.



## 5. Scelta dei Colori e Stile

* 
**Evitare i colori predefiniti:** Le palette classiche (es. blu, arancione, o verde/rosso) possono dare false interpretazioni ai lettori o apparire poco professionali.


* 
**Palette minimaliste:** Scegliere schemi di colori armoniosi e compatti (es. toni gradienti del viola e verde acqua).


* 
**Colori neutri per elementi secondari:** Utilizzare colori come il grigio per linee di riferimento e regressione, permettendo ai colori principali di spiccare senza distrazioni.



## 6. Dimensioni e Sovrapposizione degli Elementi

* 
**Visibilità dei marker e linee:** Aumentare manualmente la dimensione dei punti e lo spessore delle linee per assicurarne la leggibilità, senza renderli così grandi da causare sovrapposizioni che nascondono i dati.


* 
**Controllo dell'ordine (Z-order):** Controllare l'ordine dei livelli dell'immagine: impostare un valore gerarchico più alto per mandare i dati principali (scatter points) in primissimo piano e uno più basso per mantenere linee o curve sullo sfondo.



## 7. Struttura e Posizionamento della Legenda

* 
**Nessuna sovrapposizione:** Non usare le impostazioni predefinite che posizionano la legenda all'interno del grafico, coprendo dati importanti.


* 
**Disposizione compatta all'esterno:** Spostare la legenda all'esterno della tela (es. ancorata in alto al centro), rimuovere i bordi (frame) e riorganizzarne il layout in più colonne orizzontali per risparmiare spazio sul documento.



## 8. Post-Elaborazione e Inserimento nel Layout Finale

* 
**Software di grafica vettoriale:** Usare strumenti come CorelDraw, Adobe Illustrator o Inkscape per rifinire la figura vettoriale esportata dal codice.


* 
**Modifiche di fino:** All'interno del software è possibile rimuovere spazi bianchi invisibili, standardizzare millimetricamente le distanze degli elementi della legenda, spostare le etichette degli assi o aggiungere descrizioni personalizzate e frecce.


* 
**LaTeX vs Word:** In ambienti come LaTeX/Overleaf inserire la figura in formato PDF. Se si lavora su Microsoft Word (che non legge i file PDF), convertire e inserire l'immagine vettoriale nei formati SVG o EMF per evitare la sgranatura dell'immagine.