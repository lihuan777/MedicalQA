# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-233b1dd363774ef38fa1ac1ea4171eac", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat", # V3
    # model="deepseek-reasoner", # R1





    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": '''


        综上所述，为了充分利用数据实例内的语义信息，并构建高准确率与强相关的知识库，提出一种基于BERT构建医疗信息知识库的方法，模型架构如图4-1。DPR利用双编码器架构，通过将问题和文档段落映射到低维向量空间，实现了更加高效和精准的检索。在这一架构中，问题和文档分别经过独立的编码器处理，生成对应的语义向量，然后通过计算问题向量和文档向量之间的相似度来实现匹配。这一方法的优势在于能够通过深度学习模型捕捉到词语之间的深层语义关系，克服了传统基于词频的稀疏表示方法的局限性，提供了更为精准的检索结果。与传统的基于检索关键词匹配的技术相比，DPR在面对复杂的查询和文档时，能够有效识别文档中的关键信息，并且能够处理自然语言中复杂的语义变换。此外，DPR模型的引入大大提高了文本检索的效率，尤其在处理大规模文档库时，能够通过更高效的向量化表示减少计算复杂度，从而加快检索速度。通过结合这些现代深度学习技术，该方法在智能问答系统、搜索引擎优化以及学术文献检索等多个应用领域中，提供了更加精准和高效的解决方案。
        
        
        上面的文字是我的研究内容，但他是偏向于检索技术的，但是我的题目是基于检索技术的知识库构建，应该偏向说明知识库，针对以上信息对上文进行修改
        
        
        用一段文字不要分点。


        
        '''
        },
    ],
    stream=False
)

print(response.choices[0].message.content)