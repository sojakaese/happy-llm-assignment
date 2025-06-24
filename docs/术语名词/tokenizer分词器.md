分词器的作用是把自然语言输入切分成 token 并转化成一个固定的 index, 例如，如果我们将词表大小设为 4，输入“我喜欢你”，那么，分词器可以将输入转化成：

```
input: 我
output: 0

input: 喜欢
output: 1

input：你
output: 2
```

当然，在实际情况下，tokenizer 的工作会比这更复杂。例如，分词有多种不同的方式，可以切分成词、切分成子词、切分成字符等，而词表大小则往往高达数万数十万。
Byte Pair Encoding (BPE)、WordPiece、Unigram、SentencePiece
