# Auditoria da eleição do 26º Knesset (Israel, 27/10/2026) com dados públicos

Testes forenses nos resultados por urna (קלפי) publicados pela Comissão Central de Eleições, comparados com a eleição de 2022 (25º Knesset).

## Dados

Baixe de cada site de resultados o arquivo "Results by Voting Station" e salve em `dados/`:

| Arquivo | Origem |
|---|---|
| `dados/expb.csv` | votes26.bechirot.gov.il (2026) |
| `dados/expb_2022.csv` | votes25.bechirot.gov.il (2022) |
| `dados/expc.csv` | votes26.bechirot.gov.il (2026), "Results by locality", só para o mapa |

`dados/expb.csv` é o arquivo final, publicado pela comissão em 4/11/2026. Entre o arquivo de 2/11 e o final mudaram 78 urnas e entraram 3 lotes de envelopes duplos (lista em `dados/diff_expb_2026-11-02_2026-11-04.txt`): as 11 urnas apontadas como erro de transcrição (9 da regra de 15% e 2 da regra de 3%) foram todas corrigidas na direção prevista, e a comissão corrigiu outras 67 trocas de letras vizinhas de 1 a 27 votos que os limiares daqui não alcançavam. Ao todo 1.789 votos movidos ou acrescentados em 5,7 milhões; nenhuma cadeira muda e nenhum teste sai da segunda casa decimal. Versões anteriores: entre o arquivo de 30/10 e o de 2/11 mudaram 8 urnas (lista em `dados/diff_expb_2026-10-30_2026-11-02.txt`): 1 das 9 urnas apontadas como erro de transcrição foi corrigida (Idan 1), mais 3 deslizes do mesmo tipo que os limiares daqui não pegavam e 4 trocas de letra de até 13 votos. Efeito nacional: 155 votos de רץ para רק e 1 votante a mais; nenhuma cadeira muda. Urnas com `סמל ישוב` = 9999 são envelopes duplos (soldados, diplomatas, hospitais), sem eleitorado próprio; ficam fora dos testes de comparecimento.

`dados/levant_outlines.json` são os contornos dos países (Natural Earth 10m), usados só pelo mapa. `dados/bycode2021.xlsx` é a lista de localidades do Escritório Central de Estatística (CBS), copiada do pacote R [il.cbs.muni](https://github.com/matanhakim/il.cbs.muni). Ela dá a religião (judaica, árabe, tribo beduína, cidade mista) e o tipo de cada localidade, pelo mesmo código usado pela comissão eleitoral. Drusos, cristãos e beduínos fora das tribos são identificados por listas de localidades no próprio script.

`parties.csv` traz o nome em inglês de cada partido de 2026 pelas letras da cédula.

## Como rodar

```bash
pip install pandas numpy scipy openpyxl cvxpy

python3 forensics.py dados/expb.csv dados/boxes.pkl            # 2026: valores impossíveis, dígitos, comparecimento, Klimek, anulados
python3 sim_null.py dados/boxes.pkl                                   # 2026: dígitos e comparecimento contra simulação binomial
python3 forensics.py dados/expb_2022.csv dados/boxes_2022.pkl  # mesmos testes em 2022, como referência
python3 sim_null.py dados/boxes_2022.pkl
python3 compare_2022.py                                               # urna a urna e cidade a cidade, por blocos
python3 arab_boxes.py                                               # urnas de maioria árabe, 2022 x 2026
python3 haredi_boxes.py                                               # urnas de maioria haredi, 2022 x 2026
python3 envelopes.py                                                  # urnas de envelope duplo (soldados etc.), 2022 x 2026
python3 religious_groups.py                                           # testes por grupo religioso / comunidade, 2022 x 2026
python3 transfer.py locality                                          # transferência de votos 2022 -> 2026 (por localidade; 'box' para urnas pareadas)
python3 more_findings.py                                              # cadeiras, cenário sem o salto árabe, origem dos novos eleitores, fronteiras
python3 extra_tests.py                                                # anulados 2022 x 2026, anulados x partido, % inteiros, mudança de comparecimento por urna
python3 invalid_profile.py                                            # anulados x perfil da cidade (CBS: idade, escolaridade, nível socioeconômico)
python3 same_site.py dados/expb.csv                                   # cada urna x as outras do mesmo local; lista possíveis erros de digitação da ata
python3 fringe_check.py                                               # segundo detector de erros de transcrição (listas nanicas com muitos votos numa urna)
python3 slip_check.py dados/expb.csv                                  # detector específico do erro de linha vizinha na ata (רץ/רק etc.), limiares mais baixos
python3 committee_check.py                                            # por comissão regional (centro de digitação): dígitos e votos anulados, 2026 x 2022
python3 map_localities.py                                             # mapa: primeiro colocado e maioria de bloco por localidade (precisa de dados/expc.csv)
python3 diff_results.py dados/expb_preliminar.csv dados/expb.csv        # o que mudou entre dois downloads (preliminar x final)
```

## Notas de método

- A coluna `ריכוז` agrupa as urnas de um mesmo local de votação. `same_site.py` compara cada urna com as outras do mesmo local e procura o padrão de erro de transcrição: um partido forte no local some numa urna, e a mesma quantidade de votos aparece numa lista quase sem votos ali. A lista sai em `dados/likely_transcription_errors.csv`.
- As urnas de envelope duplo são enchidas em lotes (cerca de 600 envelopes em 2022 e 500 em 2026), por isso os dígitos do total de votantes delas não são testados.

- Teste de dígitos: a referência principal é `sim_null.py`, que refaz cada contagem como Binomial(votos válidos, fração observada) e compara o qui-quadrado observado com o simulado. A referência suavizada de `forensics.py` rejeita com frequência em partidos pequenos nos dois anos, por limitação do método.
- Os partidos mudaram entre 2022 e 2026; a comparação usa blocos: arab, haredi, netanyahu_coalition e likud.
- Só parte das urnas mantém o mesmo número entre eleições; a comparação por cidade cobre o resto.

## Correções e sugestões

eleicoes@lilianezukerman.com
