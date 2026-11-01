# Estudo: representação polinomial padronizada das árvores

## Estado da implementação

A expansão esparsa, o skeleton potencial/realizado e os coeficientes simbólicos
e numéricos estão implementados em
`src/tsfmxai/representations/canonical_polynomial.py`, com testes em
`tests/unit/test_canonical_polynomial.py`. O notebook executado
`1_skeleton_and_coefficients.ipynb` reúne 16 casos construídos com motivação
por linha, 20 árvores efetivamente sorteadas, comparação funcional em 31
históricos por caso e três exemplos de rejeição explícita. Os notebooks
seguintes documentam a cobertura dos skeletons, o treinamento dos modelos
encoder-decoder e a análise dos erros. O mapa esparso é a representação
canônica usada nesses experimentos.

Os coeficientes numéricos são calculados com aritmética racional exata **dos
valores binários fornecidos como `float`**, sem arredondamento ou limiar para
remover termos. Isso não equivale a interpretar `0.1` como o racional decimal
exato `1/10`. A apresentação converte os resultados para decimais curtos;
os valores exatos continuam disponíveis em `coefficient_exact`.

Para comparar funções realizadas, usar `canonical_key`: ela inclui somente
os monômios e coeficientes exatos, independente dos nomes dos parâmetros ou
da fatoração da entrada. A igualdade do objeto completo também considera
proveniência simbólica e configuração. A conversão para `float` sinaliza
overflow ou perda de coeficiente não nulo para zero; a forma racional continua
disponível. `format_canonical_polynomial(..., exact=True)` mostra frações
exatas; a apresentação decimal padrão é uma aproximação para leitura.

O limite de grau é verificado durante a expansão da entrada, antes de
substituir os parâmetros: `0*u^5` e termos de quinto grau que se cancelariam
também são rejeitados quando `max_degree=4`. A garantia se refere às entradas
aceitas por esse contrato. Além do número de monômios de lag (`max_terms`),
`max_coefficient_terms` controla a expansão de parâmetros dentro de cada
coeficiente. Os casos de borda e a validação independente com SymPy estão na
seção de revisão do mesmo notebook executado.

## Objetivo e relação com o gerador

O gerador continua sorteando a AST binária com seus parâmetros e atrasos. Este
experimento estuda uma **segunda representação, derivada da árvore sorteada**:
expandir os produtos sobre as somas, reunir termos iguais e emitir uma soma
esparsa em ordem fixa. A AST sorteada e seu `MechanismBinding` permanecem
registrados; a transformação não modifica o mecanismo usado para gerar séries.

Interpretamos “até quatro lags e mais uma constante” como **grau máximo quatro
em cada monômio de lags**, além do termo independente de grau zero. A constante
também pode multiplicar um monômio como coeficiente. Esta restrição vale depois
da expansão completa, não apenas para cada nó `Multiply` da árvore original.

O estudo cobre inicialmente a subgramática polinomial `Parameter`, `Lag`,
`Add`, `Multiply` de `src/tsfmxai/representations/expression.py`. A AST também
contém `Tanh`, e o gerador Shao–Wu pode introduzir `Tanh`: uma
`tanh` aplicada a uma expressão não se transforma, em geral, numa soma finita
de monômios de lags. Esses mecanismos precisam de uma futura representação
por blocos/átomos não polinomiais ou de uma classe de saída separada. Não serão
silenciosamente tratados como polinômios. A inovação aleatória é externa à
função determinística da AST; seu modelo/distribuição será preservado como
metadado, não como coeficiente de lag.

## Identidade estável das séries e dos atrasos

Usar `s_0, s_1, …` como identificadores globais imutáveis das séries do
sistema. `X` (entrada), `Y` (alvo), variável observada/oculta e disponibilidade
do futuro são **papéis de uma tarefa de avaliação**, não nomes que redefinem o
mecanismo. Assim, a mesma equação conserva a mesma chave canônica mesmo se um
experimento esconder uma série ou alterar a escolha das saídas. Uma tabela de
metadados faz a apresentação: `s_0 -> X_0`, `s_2 -> Y_0`, etc. Para sistemas
multivariados, há uma expressão por série-alvo, identificada pelo mesmo ID
global; cada termo pode referir-se a qualquer série passada, inclusive o
próprio alvo. A convenção temporal é `s_j[t-1]` = último valor disponível no
instante `t`; lags válidos são inteiros positivos.

**Átomo de lag:** `(series_id, lag)`. O `lag` é o valor inteiro obtido de
`MechanismBinding.lag_values[slot]`, não o nome do slot. Um monômio é uma tupla
ordenada desses átomos, **com repetição**: `s_0[t-1]^2` e
`s_0[t-1] s_1[t-2]` são distintos. A constante tem chave vazia `()`.

**Ordem de saída:** grau crescente (`0, 1, 2, 3, 4`); dentro de cada grau,
comparação lexicográfica das tuplas de átomos, ordenados por `series_id` e
depois por `lag`. A saída contém somente os termos presentes. Essa ordem é
uma decisão do benchmark, não uma propriedade física das séries. Versões
futuras podem escolher outra ordem, mas a versão da convenção deve acompanhar
cada amostra e cada target de treinamento.

## As duas saídas

1. **Skeleton estrutural:** sequência ordenada das chaves dos monômios com
   coeficiente resultante não nulo. É um suporte de *termos*: preserva quais
   lags aparecem juntos em interações, e não só a lista de variáveis
   participantes. Pode-se derivar dele um suporte binário de série–lag, que
   perde a informação sobre interações. Para a avaliação, decidir também se
   queremos um *suporte potencial* antes de substituir os parâmetros e um
   *suporte realizado* depois de substituí-los; valores zero podem apagá-lo.
2. **Expressão parametrizada padronizada:** a mesma sequência de chaves
   acompanhada por um coeficiente em cada posição. O armazenamento primário
   proposto é o mapa esparso `monômio -> coeficiente`, mais simples e menos
   ambíguo que uma árvore binária artificial. Um serializador determinístico
   pode reconstruir uma AST binária de soma de produtos quando um modelo ou
   visualização precisar de árvore. Há duas variantes úteis de coeficiente:
   expressão simbólica dos slots de parâmetro (para rastreabilidade) e valor
   numérico após `MechanismBinding` (para comparar funções concretas).

Exemplo, com `u=s_0[t-1]` e `v=s_2[t-2]`:

```text
AST amostrada:  (a*u + b)*(c*v + d) + e*u
Forma expandida: (b*d) + (a*d + e)*u + (b*c)*v + (a*c)*u*v

Ordem/skeleton: [(), ((0,1),), ((2,2),), ((0,1),(2,2))]
Coeficientes:   [b*d, a*d+e, b*c, a*c]
```

Por exemplo, se `a*d+e=0` após a vinculação dos parâmetros, o termo `u`
deixa de integrar o skeleton **realizado**. O termo de interação `u*v` ainda
faz `u` participar da função: suporte de termos e suporte de variáveis são
perguntas diferentes. Se dois nós `Lag` apontam para o mesmo `series_id` e
lag resolvido, devem ser reunidos, mesmo com slots distintos na árvore.

## Algoritmo de referência proposto

Uma passagem bottom-up associa a cada subárvore um mapa esparso. `Parameter`
gera o termo constante; `Lag` gera um monômio de grau um. `Add` une mapas,
somando coeficientes nas chaves iguais. `Multiply` faz a convolução entre os
mapas: multiplica coeficientes, concatena/ordena os átomos e reúne chaves
iguais. Só após consolidar o resultado se removem coeficientes exatamente
nulos; depois se emite a lista ordenada. Um limite de grau e um limite de
número de termos detectam expansão incompatível ou excessiva. **Não se pode
descartar silenciosamente termos de grau cinco**, pois isso mudaria a fórmula.

Os coeficientes atuais são `float`; equivalência algébrica exata exige cuidado
com arredondamento e ordem das somas. Uma primeira referência de testes pode
usar coeficientes racionais/exatos, seguida da conversão para `float` para
executar as séries. Para amostras reais do gerador, registrar uma política
explícita de zero/tolerância e comparar funções numericamente. Não prometer
unicidade matemática universal com `float` e limiar arbitrário. Dentro da
subgramática polinomial, com variáveis e domínio de coeficientes fixos, a
representação por coeficiente de cada monômio é a forma normal pretendida;
fora dela, a garantia não foi estabelecida.

## Caminho para o encoder–decoder

```text
AST sorteada + binding
       |-> algoritmo exato -> skeleton + mapa esparso -> target verificável
       `-> encoder hierárquico -> decoder esparso ordenado -> predição
```

O algoritmo determinístico deve produzir o target e servir de oráculo de
verificação. Um modelo pode codificar a árvore de baixo para cima e decodificar
uma sequência de termos ordenados: primeiro decide a próxima chave (ou `EOS`),
depois o coeficiente dessa chave. O decoder só pode emitir chaves maiores que
a última, impedindo duplicatas e desordem. Duas heads — suporte e coeficiente
— podem ser alternativas. Um vocabulário denso de todas as combinações de
lags cresce rapidamente com número de séries e lag máximo; por isso, comparar
representação densa, lista esparsa e geração hierárquica de chaves.

Para a tarefa **AST conhecida -> forma padronizada**, uma rede não ganha
automaticamente em velocidade nem em correção: a expansão determinística já
produz a resposta e é a baseline obrigatória. O encoder–decoder tem interesse
como representação aprendida ou como componente de um modelo que recebe
**observações da série** e precisa recuperar o mecanismo desconhecido. Tempo
de inferência só poderá ser afirmado após medir o algoritmo e a rede em CPU e
GPU, incluindo overhead e tamanho das árvores.

## Plano de execução e critérios de aceite

1. **Fixar o escopo matemático.** Formalizar grau máximo quatro, lags
   positivos, tratamento de repetição, constante, parâmetros e inovação;
   definir saída explícita para `Tanh` fora do escopo. Aceite: exemplos
   admissíveis e inadmissíveis documentados.
2. **Fixar identidade multivariada.** Criar IDs globais estáveis, metadados de
   apresentação/papel `X`/`Y`, ID da série-alvo e versão da convenção de lag.
   Aceite: ocultar uma série ou trocar seu papel não altera as chaves do
   mecanismo.
3. **Especificar chaves, ordem e serialização.** Definir chave de átomo,
   monômio com multiplicidade, grau, ordenação, `EOS` e versão do formato.
   Aceite: exemplos multivariados com lag repetido produzem a ordem esperada.
4. **Decidir semântica dos coeficientes.** Separar coeficiente simbólico,
   coeficiente numérico, suporte potencial e suporte realizado; fixar domínio
   exato para testes e política de zeros numéricos. Aceite: cancelamentos
   como `a*u - a*u` e valores nulos têm resultado especificado.
5. **Implementar a expansão esparsa de referência.** Operar sobre AST e
   binding sem alterar o sorteio; rejeitar grau >4 e controlar crescimento
   combinatório. Quando promovido, colocar em `src/tsfmxai/representations/`
   com testes em `tests/`, não só neste experimento.
6. **Gerar as duas saídas e a reconstrução.** Emitir skeleton, mapa de
   coeficientes, forma legível e AST serializada de maneira determinística.
   Aceite: reconstrução avalia a mesma função que a AST inicial no domínio
   testado.
7. **Verificar a garantia de padronização.** Testar comutatividade,
   associatividade, distributividade, termos repetidos, permuta de slots de
   lag, cancelamento e idempotência; executar busca exaustiva para árvores
   pequenas e testes baseados em propriedades para árvores maiores. Aceite:
   cada classe de equivalência polinomial exata gera o mesmo mapa; registrar
   contraexemplos e condições de validade.
8. **Quantificar custo e cobertura.** Medir número de monômios, tempo,
   memória e rejeições por grau/tamanho ao variar séries, lags e operadores;
   verificar que o limite de quatro lags ainda permite diversidade útil.
9. **Construir pares para treinamento.** Gerar múltiplas ASTs equivalentes e
   ASTs não equivalentes; guardar árvore, binding, forma normal, skeleton,
   coeficientes e sementes. Separar treino/teste por mecanismo/família de
   equivalência para evitar duplicatas vazando entre conjuntos.
10. **Definir input do modelo.** Comparar encoder hierárquico da AST conhecida
    (estudo de transformação simbólica) com encoder de séries observadas
    (recuperação do mecanismo). Não misturar resultados das duas tarefas.
11. **Definir decoder esparso.** Testar emissão monotônica de chaves e
    regressão de coeficientes contra heads de suporte/coeficiente; definir
    perdas para suporte, parâmetros e semântica da função. Aceite: saída
    parseável, ordenada, grau <=4, sem chaves repetidas.
12. **Avaliar modelos e custo.** Medir exatidão do skeleton, erro de
    coeficientes, equivalência funcional em históricos novos, generalização
    para mais séries/lags e latência CPU/GPU; comparar sempre com o algoritmo
    exato e com uma representação prefixa simples.
13. **Promover o que for validado.** Versionar formato, contratos, testes,
    documentação e benchmarks reproduzíveis no lugar previsto pela
    arquitetura do repositório. Somente então considerar a forma padronizada
    como target oficial para recuperação simbólica.

## Referências e evidência

- [D’Ascoli et al. (2022), *Deep symbolic regression for recurrent sequences*](https://proceedings.mlr.press/v162/d-ascoli22a/d-ascoli22a.pdf): referência para geração de relações de recorrência como árvores e aprendizado de expressões a partir de sequências; não estabelece esta forma polinomial proposta.
- [Documentação oficial do SymPy, expansão](https://docs.sympy.org/latest/tutorial/simplification.html): expansão polinomial como soma de monômios; útil como oráculo independente nos testes.
- [Documentação oficial do SymPy, polinômios](https://docs.sympy.org/latest/modules/polys/basics.html): reunião de monômios com mesmos expoentes em termos com coeficientes.

Este arquivo é um plano de experimento, sem métricas executadas. A
implementação, os testes e seus resultados serão registrados quando feitos.
