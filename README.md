# OFDM Estimation Techniques
TP de Comunicaciones Digitales I - FIUBA

## Listo
- Simular muchos bits
- Convertir a simbolos 16-QAM
- Modular OFDM
- Canal con atenuacion constante, plano en frecuencia, con ruido gausiano
- Estimación LS del canal
- Estimacion de simbolos OFDM invirtiendo el canal
- Demodulacion, obtener bits y contar taza de error

## TO-DO
- Reimplementar canal fading con sinc que modula
- Ver como anda el LS con ese canal
- Remplementar MMSE
- Ver como funciona waterfilling: es una preecualizacion en el transmisor, preamplifica las subportadoras para las cuales el canal mas atenua, esto requiere conocimiento de la rta del canal de parte del transmisor. Seguramente el receptor le tiene que transmitir el canal de algun modo(algun piloto, un ping pong o ver que lo manden cuando hacen sincronizacion).

## Referencias
- modelos matematicos en las refs del informe
- van der Beek: la papa. De ahi sale todo lo basico.
- del Proakis ver seccion de fading channels (seccion 13.6 Multicarrier Modulation)