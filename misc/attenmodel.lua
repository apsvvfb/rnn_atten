require 'nngraph'
require 'nn'
local attenmodel = {} 
function attenmodel.subatten(seq_len,feat_dim,hidden_size,batch_size)
	local inputs = {}
	local outputs = {}

	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())

	local feats = inputs[1] 	-- [batchsize x seqlen x featdim]
	local preHidden = inputs[2]	-- [batchsize x hiddensize]

	-- PAPER: Hierarchical Recurrent Neural Encoder for Video Representation with Application to Captioning
	-- input: x_1, x_2, ..., x_n
	-- e_i^(t)=W * tanh( W_a * x_i + U_a * h_{t-1} + b_a)
	-- alpha_i^(t) = exp(e_i^(t)) / sum from {j=1} to {n} {exp(e_j^(t))}
	-- output: sum from {i=1} to {n} { alpha_i^(t) * x_i }
local considerZero = 0
local KnowSeq = 1

featsplit = nn.SplitTable(2)(feats)
--e_out1 = W_a * x_i
local e_out1 = nn.MapTable():add(nn.LinearNoBias(feat_dim, 1))(featsplit)
--e_out2 = U_a * h_{t-1} + b
local e_out2 = nn.Linear(hidden_size,1)(preHidden)
local e_out3
if KnowSeq == 1 then
        local e_out1_tensor = nn.Reshape(seq_len,batch_size,1)(nn.JoinTable(1)(e_out1))
        local e_out2_rep = nn.Reshape(seq_len,batch_size,1)(nn.Replicate(seq_len)(e_out2))
        if considerZero == 0 then
                --e_out3 = e_out1 + e_out2 = W_a * x_i + U_a * h_{t-1} + b
                e_out3 = nn.SplitTable(1)(nn.CAddTable()({e_out1_tensor,e_out2_rep}))
        elseif considerZero == 1 then
                --e_out3_zero = e_out1 + e_out2*e_out1
                local e_out2_e_out1 = nn.CMulTable()({e_out1_tensor,e_out2_rep})
                e_out3 = nn.SplitTable(1)(nn.CAddTable()({e_out1_tensor,e_out2_e_out1}))
        end
        --[[
        --e_i=W * tanh( e_out3)
        local model_ei = nn.Sequential()
                :add(nn.Tanh())
                :add(nn.LinearNoBias(1,1))
        local e_i = nn.MapTable():add(model_ei)(e_out3)
        --e_sum = sum from {i=1} to {n} {exp(e_i)}
        local e_sum = nn.CAddTable()(nn.MapTable():add(nn.Exp())(e_i))
        --alpha_i
        local e_sum_rep = nn.Reshape(seq_len,batch_size)(nn.Replicate(seq_len)(e_sum))
        local e_i_tensor = nn.Reshape(seq_len,batch_size)(nn.JoinTable(1)(e_i))
        local alpha_i = nn.CDivTable()({e_i_tensor,e_sum_rep})
        --]]
else    
        --e_out3 = e_out1 + e_out2 = W_a * x_i + U_a * h_{t-1} + b
        e_out3 = nn.MapTable():add(nn.AddTensor(e_out2))(e_out1)
end
--e_i=W * tanh( e_out3)
local model_ei = nn.Sequential()
        :add(nn.Tanh())
        :add(nn.LinearNoBias(1,1))
local e_i = nn.MapTable():add(model_ei)(e_out3)
--alpha_i = exp(e_i^(t)) / sum from {j=1} to {n} {exp(e_j^(t))} = softmax(e_i)
e_i_tensor = nn.JoinTable(2)(e_i)
alpha_i = nn.SoftMax()(e_i_tensor)
alpha_i = nn.Transpose({1,2})(alpha_i)
local alpha_i_rep = nn.Transpose({1,3})(nn.Replicate(feat_dim)(alpha_i))
--out_i = alpha_i * x_i 
local out_i = nn.CMulTable()({feats,alpha_i_rep})
--out = sum from {i=1} to {n} {out_i}
local out = nn.CAddTable()(nn.SplitTable(2)(out_i))

	table.insert(outputs,out)


	return nn.gModule(inputs,outputs)
	
end

return attenmodel
