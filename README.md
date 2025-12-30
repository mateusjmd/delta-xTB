![Banner](streamlit/media/banner.png)

# $\Delta-\mathrm{xTB}$

O $\Delta-\mathrm{xTB}$ é uma aplicação desenvolvida como projeto final da disciplina de *Machine Learning* do curso de Bacharelado 
em Ciência e Tecnologia da [**Ilum — Escola de Ciência**](https://ilum.cnpem.br). Seu objetivo central é empregar métodos supervisionados de *Machine 
Learning* para a predição de propriedades termodinâmicas e eletrônicas de espécies químicas a partir de sua representação estrutural
em **[SMILES](https://pubs.acs.org/doi/10.1021/ci00057a005)**.

A proposta insere-se no cenário contemporâneo da química computacional, no qual a ampliação do espaço químico explorável — 
tanto em diversidade estrutural quanto em complexidade eletrônica — impõe desafios significativos em termos de custo computacional, 
escalabilidade e tempo de resposta. Nesse contexto, torna-se essencial o desenvolvimento de abordagens que conciliem rigor 
físico-químico, eficiência numérica e viabilidade computacional, sem comprometer a confiabilidade das predições.

A versão mais recente da aplicação pode ser acessada em: [**Δ-xTB**](https://delta-xtb.streamlit.app/)


## Execução Local

Para rodar o $\Delta-\mathrm{xTB}$ localmente, siga as etapas:

1. Instale o [**Python 3.12.7**](https://www.python.org/downloads/).
2. Clone o repositório.
3. Instale as bibliotecas necessárias executando o seguinte comando com o terminal no diretório raiz do repositório:

  ```python
  pip install -r requirements.txt
  ```
4. Ainda no terminal, navegue até diretório `streamlit` e execute o comando:
  ```python
  python -m streamlit run main.py
  ```
5. Por fim, a interface do Streamlit abrirá no navegador padrão automaticamente.


> **📌 Observação:** A execução local da aplicação é suportada nos sistemas operacionais Windows e Linux.


## Fluxo de Acesso

Para compreender os aspectos metodológicos envolvidos no desenvolvimento desse projeto, acesse na ordem:

1. Pré-processamento do *dataset* `QM9` no diretório `dataset_processing`.
2. Modelos de *Machine Learning* no diretório `machine_learning_models`: cada subdiretório destina-se ao conjunto de arquivos referente ao respectivo modelo, contendo um *Jupyter Notebook* com detalhes técnicos, um *script* em Python para o *tunning* do modelo utilizando HPC (*High Performance Computing*) e os estudos de otimização de hiperparâmetros obtidos pelo Optuna.
3. Aplicação final à interface gráfica utilizando o Streamlit, com integração entre *back* e *front-end* no diretório `streamlit`.


## Desenvolvedores

Esse projeto foi desenvolvido de maneira independente por Mateus de Jesus Mendes, a partir de um *fork* do projeto inicialmente concebido conjuntamente com [**Edélio G. M. de Jesus**](https://github.com/EdelioGabriel) e [**Matheus P. V. da Silveira**](https://github.com/Velky2), mantendo-se a autoria das devidas contribuições reutilizadas para o seu desenvolvimento. A versão inicial pode ser acessada em: [**R2D2**](https://github.com/Velky2/R2D2).


## Professor Orientador

Esse projeto foi orientado e fundamentado teoricamente a partir da disciplina de Aprendizado de Máquina lecionada pelo [**Prof. Dr.
Daniel Roberto Cassar**](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K4262774J5).

![Footer](https://ilum.cnpem.br/wp-content/uploads/2023/01/Ilum_800px-1536x287.png)
