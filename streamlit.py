import streamlit as st
import pandas as pd


st.title("Implementação da TRI no CASD :airplane_departure:")
st.header("Mas quem é o :red[maluco] que decidiu fazer isso?")
st.caption("_calma_, _calma_, nem sabemos se deu certo ainda kkk")
st.write("Ah, o Brunin Pimentel (vulgo braceta) tava sem nada pra fazer na semaninha e decidiu entender o que aquela formula quer dizer...")
st.write('''A ideia dessa exposição não é ser detalhada, 
já que muitos autores já fizeram esse favor pra nós. É apenas uma explicação
 por cima do que quer dizer tudo isso. Confia que :red[não] é tão agradável entender
   esses trem um pouco mais profundamente.''')
st.header("Mas oq é esse trem de TRI? :thinking_face:")
st.write('''
Em minhas palavras, a TRI é uma teoria da psicometria que tem como base métodos estátisticos para determinar traços latentes
dos indivíduos. 
Como assim traços latentes?
Um exemplo seria a proficiência ou o conhecimento/inteligência de um aluno ao lidar com matemática. Ok, mas como poderíamos medir isso?
A ideia principal aqui é estabelecer parâmetros relevantes que não seja: Ah, o aluno acertou uma questão que ninguém acertou, então ele é brabo.
Na verdade, vamos ver que isso pode por vezes até ser ruim!
A TRI vai buscar medir habilidades que não podemos ver, a exemplo da inteligência ou conhecimento do aluno na matéria. Nesse modelo, temos três parâmetros
principais: a, b e c, que serão discutidos :blue[AGORA].
''')
st.header("Os :green[três] parâmetros :shocked_face_with_exploding_head:")
st.caption('''Deixando claro que muitas das ideias aqui escritas foram retiradas de um pdf disponibilizado pelo IMPA, com 
autoria de Deivison de Albuquerque da Cunha, cujos estudos também foram muito baseados segundo Andrade, Tavares e Valle
           ''')
st.write(r"""
i. parâmetro de discriminação (a) – consiste na aptidão do item em
distinguir indivíduos com habilidades diferentes;

ii. parâmetro de dificuldade (b) – trata-se da habilidade mínima que
um respondente precisa para ter uma grande probabilidade de dar a
resposta correta;

iii. parâmetro de acerto ao acaso (c) – é a probabilidade de um
respondente com baixa proficiência responder corretamente um item.
""")
st.header("Como faço pra calcular esses parâmetros em uma dada questão?")
st.write("""Isso é um desafio bem grande. Os valore dos parâmetros a, b e c são calculados através de pré-testagens
(calibragem dos itens), utilizando o método da máxima verossimilhança. Isso não dá pra fazer na mão e precisamos
de um programa que o faça. Em Python tem uma biblioteca chamada pyirt que em tese estima esses valores, mas a adaptação
dela no código final não ficou tão boa e a estimativa desses parâmetros pra cada questão acabou sendo feita
na linguagem R, a qual pode ser vinculada a uma bilbioteca chamada mirt que faz quase tudo pra gente. Primeiramente,
 precisamos das respostas dos alunos pra cada questão em um DataFrame de modo que cada questão certa recebe atribuição 1
  enquanto cada questão errada recebe atribuição 0. Um arquivo desse tipo é explicitado a seguir:""")
df_respostas = pd.read_csv('TRIcasd.csv')
st.write(df_respostas)
st.title(":red[A estimativa]")
st.write('''Tive a saída de estimar esses valores em R. Então tiramos a primeira linha do dataframe e em seguida a
 primeira coluna, após isso rodamos a função em R:''')
code = '''library(mirt) 
mod3 <- mirt(TRIcasd, 1, itemtype='3PL')
coef(mod3, simplify=TRUE, IRTpars=TRUE)'''
st.code(code, language='r')
st.write(''' Isso vai printar os valores estimados de a, b e c. Como a bilbioteca é orinalmente em inglês, há somente a
troca de c por g, o qual por ser o fator de chute tem significado _guess_. Com esses valores montamos um arquivo csv cujas 
colunas são a, b e g de cada questão:
''')
df_questoes = pd.read_csv('questoes.csv')
st.write(df_questoes)
st.write("Depois disso precisamos implementar esses valores no algoritmo pra calcular a proficiência do mesmo modo que o ENEM faz.")
st.write("Pra fazer isso, precisamos entender o modelo 3PL ou ML3:")
st.title("O Modelo Logístico Unidimensional de :blue[3 Parâmetros] (ML3)")
st.latex(r'''
P\left(U_{ij}=1|\theta_j\right)=c_i + (1-c_i)\cdot\frac{1}{1+e^{-Da_i\left(\theta_j-b_i\right)}}
''')
st.markdown(r"""
$U_{ij}$ é uma variável dicotômica que assume os valores 1, quando o
indivíduo j responde corretamente o item i, ou 0 quando o indivíduo j não
responde corretamente ao item $i$.

$\theta_j$ representa a habilidade (traço latente) do $j$-ésimo indivíduo

$P\left(U_{ij}=1|\theta_j\right)$ é a probabilidade de um indivíduo $j$ com habilidade
responder corretamente o item $i$ e é chamada de Função de Resposta do
Item – FRI.

$b_i$ é o parâmetro de dificuldade (ou de posição) do item $i$, medido na
mesma escala da habilidade.

$a_i$ é o parâmetro de discriminação (ou de inclinação) do item $i$, com
valor proporcional à inclinação da Curva Característica do Item — CCI no
ponto $b_i$.

$c_i$ é o parâmetro do item que representa a probabilidade de indivíduos
com baixa habilidade responderem corretamente o item $i$ (muitas vezes
referido como a probabilidade de acerto casual).

$D$ é um fator de escala, constante e igual a 1. Utiliza-se o valor 1,7
quando deseja-se que a função logística forneça resultados semelhantes ao
da função ogiva normal.
""")
st.header("Mas isso :black[não] basta... :sweat_smile:")
st.write('''Agora devemos pegar essa função probabilidade e estimar a probabilidade de acerto de cada aluno através
 de uma função de verossimilhança. Depois disso a proficiência tem como base o mínimo dessa funçao, a qual deve ser derivada.
 A estimativa ainda leva em conda a média das proficiências e o desvio padrão, **:red[parte]** do código fica assim:
''')
code = '''# Função que calcula a probabilidade de acerto em uma questão
def prob_acerto(theta, a, b, g):
    return g + (1 - g) * np.exp(a * (theta - b)) / (1 + np.exp(a * (theta - b)))

# Função de verossimilhança
def verossimilhanca(theta, respostas, a, b, g):
    p = prob_acerto(theta, a, b, g)
    return np.prod(p ** respostas * (1 - p) ** (1 - respostas))

# Função para estimar a proficiência de um aluno
def estimar_proficiencia(respostas_aluno, a, b, g):
    result = minimize_scalar(lambda theta: -verossimilhanca(theta, respostas_aluno, a, b, g))
    return result.x

# Estimação das proficiências dos alunos
proficiencias = []
for i, aluno in enumerate(alunos):
    prof = estimar_proficiencia(respostas[i], a, b, g)
    proficiencias.append(prof)

# Média e desvio padrão dos parâmetros b das questões
theta_medio = np.mean(b)
theta_desvio = np.std(b)

# Escala de proficiência Enem (0 a 1000)
prof_enem = (500 + 100 * (np.array(proficiencias) - theta_medio) / theta_desvio).round(2)

# Criação do arquivo de saída
df_saida = pd.DataFrame({'aluno': alunos, 'proficiencia': prof_enem})
df_saida.to_csv('saida.csv', index=False)'''
st.code(code, language='python')
st.write(''' No final temos um arquivo csv de saída gerado, o qual possui os estudantes com suas proficiências
''')
df_saida = pd.read_csv('saida.csv')
st.write(df_saida)
st.write("""Nesse caso deu um valor doido pra Ana Beatriz por dois fatores: numero reduzido de questões e número reduzido 
         de alunos para o teste.""")
st.header("E sempre lembremos...")
st.title("26 é BIXO")


