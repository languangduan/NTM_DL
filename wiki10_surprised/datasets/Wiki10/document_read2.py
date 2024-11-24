from lxml import html


#z这一段代码可以提取该节点之中最靠前的一段连续的p节点中的所有文字
#!!!!!注意有些文章可能没有摘要，或者不包含正经单词，因此如果要拆分出tokens的话，需要确认长度

def get_node_content(node):
    """
    递归获取节点及其所有子节点的文本内容。
    :param node: lxml 节点
    :return: 合并后的字符串内容
    """
    content = ""
    
    # 获取当前节点的文本内容
    if node.text:
        content =content+' '+ node.text.strip()
    
    # 遍历所有子节点并递归获取它们的内容

    for child in node:
        content =content+' '+ get_node_content(child)  # 递归调用，获取子节点内容
    
    # 获取当前节点尾部的文本内容（如果有）
    if node.tail:
        content =content+' '+ node.tail.strip()
    
    return content



def fetch_content_with_xpath(url, xpath= "/html[1]/body[1]/div[1]/div[1]/div[1]/div[2]"):
    """
    使用 XPath 从网页提取内容。
    :param url: 目标网页的 URL
    :param xpath: 要查找的 XPath 表达式
    :return: 提取的节点
    """
    start_point=0
    try:
        # 获取网页内容
        with open(url, "r", encoding="utf-8") as file:
            html_content = file.read()

        # 使用 lxml.html 解析内容
        tree = html.fromstring(html_content)

        # 使用 XPath 查找内容
        result = tree.xpath(xpath)[0]
        #接下来寻找其概要，也即一直搜索p节点，直到遇到第一个不是p节点的地方
        content=''
        for child in result:
            if (child.tag == "p") & (start_point==0):  # 如果是 <p> 标签，提取文本
                start_point=1
                content=content+get_node_content(child)
            elif (child.tag != "p") & (start_point==1) :
                break  # 如果开始之后遇到非 <p> 标签，停止遍历
            elif (child.tag == "p") & (start_point==1):
                content=content+get_node_content(child)

        return content

    except :
        pass


def document_read(file_path):
    return fetch_content_with_xpath(file_path)



# 示例用法
if __name__ == "__main__":
    path = "wiki10+_documents/documents/0a6b6211b28832d60c38846f69dc2840"  # 替换为你的目标 URL
    print(document_read(path))

