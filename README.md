# Tema 2 ASC - CUDA Proof of Work

Nume: Balica Andrei-Sebastian
Grupa: 332CD

## Organizare

Tema presupune implementarea in CUDA a doua functii din logica unui miner Bitcoin simplificat: constructia arborelui Merkle si gasirea nonce-ului prin Proof of Work.

**construct_merkle_root** — reducere binara a hash-urilor pe GPU in doi pasi:

1. `sha256_transaction_kernel`: fiecare thread calculeaza SHA256 pentru o tranzactie, toate in paralel.
2. Bucla host cu doua kerneluri intercalate in functie de dimensiunea input-ului:
   - cat timp `n > THREADS_PER_BLOCK`: `pair_combine_kernel` — fiecare thread combina exact o pereche de hash-uri (un nivel din arbore per lansare). Gestioneaza corect `n` impar prin duplicarea ultimului element.
   - cand `n <= THREADS_PER_BLOCK`: `combine_hashes_kernel` — un singur bloc reduce tot arborele ramas in shared memory, fara trafic suplimentar la global memory.

Motivul pentru cele doua kerneluri: `combine_hashes_kernel` e mai rapid cand tot input-ul incape intr-un bloc (acces exclusiv prin shared memory), dar pentru `n` mare nu poate produce un rezultat corect decat daca fiecare bloc lucreaza pe exact o pereche, de aceea se foloseste `pair_combine_kernel` in primele runde.

Race condition-urile sunt evitate prin `__syncthreads()` intre fiecare runda de combinare, niciun thread nu citeste rezultatele rundei curente inainte ca toate thread-urile sa fi terminat de scris. Rezultatele fiecarei runde sunt scrise intr-un buffer separat (double buffering), iar pointerii sunt swapati pe host dupa fiecare lansare, eliminand nevoia de copiere intre runde.

**find_nonce** — cautare cu early exit pe grupe de nonce-uri:

La fiecare iteratie, CPU-ul lanseaza un kernel cu multiProcessorCount * THREADS_PER_BLOCK nonce-uri de testat, alegere care umple toate SM-urile simultan si se adapteaza automat la orice GPU. Marimea grupei e un compromis: grupe mici inseamna mai multe verificari din partea CPU dar oprire mai rapida; grupe mari reduc overhead-ul dar risca sa calculeze nonce-uri inutile dupa ce solutia a fost gasita.

La inceputul fiecarui kernel, continutul blocului e incarcat o singura data in shared memory, astfel toate thread-urile il citesc de acolo in loc sa acceseze global memory de THREADS_PER_BLOCK ori. Fiecare thread verifica apoi un flag global marcat volatile si iese imediat daca altcineva a gasit deja solutia, fara sa mai calculeze SHA256. Volatile previne caching-ul valorii in registri, astfel thread-urile vad mai repede ca solutia a fost gasita. Odata lansat, un kernel nu poate fi oprit din exterior, flag-ul e singurul mecanism prin care blocurile dintr-un lot se pot semnaliza reciproc.

Dupa fiecare lot, CPU-ul verifica flag-ul si opreste bucla imediat daca s-a gasit un rezultat.

## Implementare

Intregul enunt este implementat: `construct_merkle_root` si `find_nonce` in `src/utils.cu`.

Dificultati intampinate:
- Dimensionarea corecta a kernelurilor astfel incat GPU-ul sa fie utilizat la capacitate fara a genera coada de blocuri in asteptare — prea putine blocuri lasa SM-uri inactive, prea multe creeaza presiune pe scheduler si cresc latenta pentru ambele functii.
- Incadrarea in timpii impusi pentru Merkle a necesitat un kernel hibrid: pair_combine_kernel pentru n mare si combine_hashes_kernel cu shared memory pentru n mic, varianta cu un singur kernel fiind la limita pragului.
- Balansarea numarului de blocuri lansate cu munca per thread in find_nonce — prea putine blocuri inseamna overhead mare din lansari repetate, prea multe calculeaza nonce-uri inutile dupa gasirea solutiei.


## Resurse utilizate

- Laboratoarele ASC — notiuni despre lansare kernel, shared memory, sincronizare 
- Gemeni — asistent pentru discutii tehnice pe baza implementarii
