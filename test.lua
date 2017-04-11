require 'nn'
require 'os'
require 'rnn'
batchsize = 3
seq = 5
featdim = 1
feats=torch.rand(3,5,1)
--feat1 = nn.SplitTable(1)(feats)
--print(feat1)
feat2 = nn.SplitTable(2)(feats)
print(feat2)
for i = 1,#feat2 do
	print(feat2[i])
end
tmp3 = nn.JoinTable(2)(feat2)
print(tmp3)
tmp4 = nn.Transpose({1,2})(tmp3)
print(tmp4)
--tmp1 = nn.SplitTable(1)(tmp3)
--tmp2 = nn.SplitTable(2)(tmp3)
--print(tmp2)
