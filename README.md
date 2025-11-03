# $R^2\text{D}^2$
> O $R^2\text{D}^2$ é um trabalho que calcula com alta precisão e velocidade a energia total de uma molécula a partir de seu SMILES

<!------------------------------------>
<img width="800" height="450" alt="Brasão_R2D2" src="https://github.com/user-attachments/assets/26b1dcfe-39a3-423b-9bcc-44307bcb0799" />

### Shut up and Calculate


## 🔎 Sumário


- [Sumário](#🔎-sumário)
- [Descrição](#-descrição)
- [Como rodar?](#como-rodar-em-seu-pc)
- [Professor](#-professor-responsavel)
- [Colaboradores](#-colaboradores)

<!------------------------------------>

## 📝 Descrição
#### ❓O que é o RDKit?
##### RDKit é uma biblioteca open-source em Python voltada para química computacional e quimioinformática. Permite criar, visualizar e manipular estruturas moleculares, calcular descritores e realizar buscas por similaridade. É amplamente usada em aprendizado de máquina aplicado a química e em pipelines de descoberta de fármacos e materiais.
------
#### ❓O que é o XTB?
##### xTB (extended Tight Binding) é um método semiempírico de química computacional desenvolvido para calcular rapidamente energias, geometrias e propriedades moleculares com boa precisão e baixo custo computacional. Ele é amplamente usado em otimizações geométricas, triagens de grandes conjuntos de moléculas e como aproximação inicial para métodos quânticos mais caros, como DFT.
----
#### ❓O que é o QM9?
##### QM9 é um banco de dados público contendo propriedades físico-químicas de cerca de 134 mil pequenas moléculas orgânicas calculadas via teoria do funcional da densidade (DFT). Inclui dados como energia total, dipolo, polarizabilidade e entalpias. É amplamente utilizado para treinar e avaliar modelos de machine learning em química computacional.
---------------------
#### ⁉️Qual o problema então?
##### O XTB é um método rápido, porém com precisão não aceitável em muitos casos, precisando-se recorrer a DFTs mais pesados computacionalmente, o que acarreta em horas para um único cálculo.
-------------

#### ❗Como vamos resolver esse problema?
##### Usaremos o $\delta$-Learning, ou seja, nosso modelo predirá a diferença entre o resultado preciso, do QM9, e o rápido, do XTB, assim, ao final calculamos a energia total como: E = XTB + $\delta$, obtendo o resultado com alta velocidade e acuracia.
-----------------

## Como rodar em seu PC
### 👨‍💻 Rodando Localmente
📋 Pré-requisitos:

1. Instale o Python em seu computador. Recomendamos a versão 3.12.7.
2. Instale todas as bibliotecas necessárias. No VSCode, use o comando:
```python
pip install -r requirements.txt
```

▶️ Executando a aplicação
1. Digite em seu terminal:
```python
cd streamlit
python -m streamlit run main.py
```
2. O aplicativo deve abrir em seu navegador padrão automaticamente.
--------------------
## 🌊 Fluxo
Caso deseje entender o trabalho, recomendamos seguir esta ordem para melhor experiência:
1. Leitura do README
-Em que é feita uma introdução geral ao trabalho 
2. Modelos de Machine Learning
-O ponto chave do projeto, pode seguir qualquer ordem para ler os seis principais, mas o Ensemble deve ser o último.
3. Streamlit
-O final do projeto, o resultado final do que foi feito.
--------------------

## 👨‍🏫 Professor responsavel

<table>
  <tr>
    <td align="center">
      <a href="#" title="Prof. Daniel R. Cassar">
        <img src="https://avatars.githubusercontent.com/u/9871905?v=4" width="100px;" alt="Foto do Daniel do Github"/><br>
          <a href="https://github.com/drcassar"><b>Prof. Dr. Daniel R. Cassar<b></a>
      </a>
    </td>
  </tr>
</table>

## 🤝 Colaboradores

<table>
  <tr>
    <td align="center">
      <a href="#" title="Edélio G. M. de Jesus">
        <img src="https://avatars.githubusercontent.com/u/208799633?v=4" width="100px;" alt="Foto do Edélio do Github"/><br>
          <a href="https://github.com/EdelioGabriel"><b>Edélio G. M. de Jesus<b></a>
      </a>
    </td>
    <td align="center">
      <a href="#" title="Mateus de Jesus Mendes">
        <img src="https://avatars.githubusercontent.com/u/210257411?v=4" width="100px;" alt="Foto do Mateus do Github"/><br>
          <a href="https://github.com/mateusjmd"><b>Mateus de Jesus Mendes<b></a>
      </a>
    </td>
    <td align="center">
      <a href="#" title="Matheus P. V. da Silveira">
        <img src="https://avatars.githubusercontent.com/u/192454172?v=4" width="100px;" alt="Foto do Matheus do Github"/><br>
          <a href="https://github.com/Velky2"><b>Matheus P. V. da Silveira<b></a>
      </a>
    </td>
  </tr>
</table>

### 💪 Como cada colaborador contribuiu?

> Edélio G. M. de Jesus: Desenvolveu os modelos Suport Vector Machine e Kernel Ridge Regression, além de fazer a análise SHAP.

> Mateus de Jesus Mendes: Atuou para integrar RDKit-XTB-QM9, além de desenvolver o modelo NGBoost e ajudar no streamlit.

> Matheus P. V. da Silveira: Desenvolveu os modelos Elatic Net, Extreme Gradient Boosting e Ensemble, além de ajudar no streamlit.





![alt text](https://ilum.cnpem.br/wp-content/uploads/2023/01/Ilum_800px-1536x287.png "Logo da Ilum completa")

<!------------------------------------>
