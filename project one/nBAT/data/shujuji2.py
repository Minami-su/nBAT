



import csv
import re
def seq2kmer0(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    # kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmer = [seq[x:x + k] for x in range(0, len(seq) + 1 - k, 3)]
    kmers = " ".join(kmer)
    return kmers
def seq2kmer1(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    # kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmer = [seq[x:x + k] for x in range(1, len(seq) + 1 - k, 3)]
    kmers = " ".join(kmer)
    return kmers
def seq2kmer2(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    # kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmer = [seq[x:x + k] for x in range(2, len(seq) + 1 - k, 3)]
    kmers = " ".join(kmer)
    return kmers
def seq2kmer3(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    # kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmer = [seq[x:x + k] for x in range(3, len(seq) + 1 - k, 6)]
    kmers = " ".join(kmer)
    return kmers
def seq2kmer4(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    # kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmer = [seq[x:x + k] for x in range(4, len(seq) + 1 - k, 6)]
    kmers = " ".join(kmer)
    return kmers
def seq2kmer5(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    # kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmer = [seq[x:x + k] for x in range(5, len(seq) + 1 - k, 6)]
    kmers = " ".join(kmer)
    return kmers
# 打开文件
with open(r"H:\parser-main\mRNA与ncRNA数据\CPPred_neg_data.fa", 'r') as f:
    # 读取文件内容
    lines = f.read()
# 使用正则表达式提取基因序列

data_list = lines.split("\n")

# 排除开头有">"的行
gene_sequences = [line for line in data_list if not line.startswith(">")]

#gene_sequences = re.findall(r'>Lncipedia\d+\n([A-Z]+)', data)
# 将基因序列写入CSV文件
with open(r"H:\parser-main\data\biio\LNCipedia_neg2.csv", 'w', newline='') as f:
    # 创建CSV写入器
    writer = csv.writer(f)
    # 写入标题行
    writer.writerow(['source', 'target'])
    #writer.writerow(['source', 'target'])
    # 循环写入基因序列
    sum = 0
    y=0
    ke=0
    for gene_sequence in gene_sequences:
        # if len(gene_sequence) <= 200:
        #     continue
        #sum += len(gene_sequence)
        # 对基因序列进行6-kmer分词
        k = 3
        #gene_sequence = list(gene_sequence)
        kmer_string0 = seq2kmer0(gene_sequence, k)
        kmer_string1 = seq2kmer1(gene_sequence, k)
        kmer_string2 = seq2kmer2(gene_sequence, k)
        # kmer_string3 = seq2kmer3(gene_sequence, k)
        # kmer_string4 = seq2kmer4(gene_sequence, k)
        # kmer_string5 = seq2kmer5(gene_sequence, k)

        x = kmer_string0.split(' ')
        # print('2',len(x))

        # rint(kmer_string)
        if len(x) >= 2000:
            ke += 1
        sum += len(x) * 3
        y += 3
        # # 写入kmer字符串到CSV文件
        #writer.writerow([kmer_string0, kmer_string1, kmer_string2,0])
        writer.writerow([kmer_string0, 0])
        writer.writerow([kmer_string1, 0])
        writer.writerow([kmer_string2, 0])
        # writer.writerow([kmer_string3, 0])
        # writer.writerow([kmer_string4, 0])
        # writer.writerow([kmer_string5, 0])
        #gene_sequence = " ".join(gene_sequence)

        #writer.writerow([gene_sequence, 0])
    print(ke/3)
    print(y)
    print(sum/y)