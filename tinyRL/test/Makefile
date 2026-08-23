# Test host dei kernel dell'algoritmo. I sorgenti di Core/Src usati qui non
# toccano l'HAL (le funzioni UART stanno in uart.c, fuori da questa lista):
# bastano due header stub vuoti per compilarli in nativo, cosi' forward/
# backward/Adam si verificano senza flashare la scheda.
#
#   make            i due test
#   make asan       gli stessi sotto AddressSanitizer + UBSan
#   make clean

CC      ?= gcc
INC      = -I stub -I ../Core/Inc
CFLAGS   = -O2 -Wall -Wextra $(INC)
SRC      = ../Core/Src/dense_layer.c ../Core/Src/neural_net.c \
           ../Core/Src/dqn.c ../Core/Src/rng.c
LDLIBS   = -lm

.PHONY: all asan clean
all: test_equiv test_update
	./test_equiv
	./test_update

test_equiv: test_equiv.c $(SRC)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

test_update: test_update.c $(SRC)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

# I "leak" segnalati sono le reti mai liberate: sul micro vivono per sempre.
# Quello che conta e' l'assenza di accessi fuori dai limiti nell'aritmetica
# delle arene (una allocazione per layer, una per il replay buffer).
asan: test_equiv.c test_update.c $(SRC)
	$(CC) -O1 -g -fsanitize=address,undefined -w $(INC) test_equiv.c $(SRC) $(LDLIBS) -o test_equiv_asan
	$(CC) -O1 -g -fsanitize=address,undefined -w $(INC) test_update.c $(SRC) $(LDLIBS) -o test_update_asan
	ASAN_OPTIONS=detect_leaks=0 ./test_equiv_asan
	ASAN_OPTIONS=detect_leaks=0 ./test_update_asan

clean:
	$(RM) test_equiv test_update test_equiv_asan test_update_asan
