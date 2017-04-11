require 'rnn'
require 'os'
require 'nn'
require 'optim'
require 'misc.attenLSTM'
printout=1
-- model parameters
attOpt = {} 
attOpt.feat_dim = 20
attOpt.hidden_size = 10
attOpt.batch_size = 7 
attOpt.seq_length = 5
attOpt.class_num = 10

model_attenLSTM = nn.attenLSTM(attOpt)
x, dl_dx = model_attenLSTM:getParameters()
local criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1)
---------------input
--inputs = {}
targets = {}
outputs, err = {}, 0
for seq = 1,attOpt.seq_length do
	--table.insert(inputs,torch.randn(attOpt.batch_size,attOpt.feat_dim))
	labeltemp = torch.Tensor(attOpt.batch_size):fill(seq)
	table.insert(targets, labeltemp)
end
inputs=torch.rand(attOpt.batch_size,attOpt.seq_length,attOpt.feat_dim)
--------------forward
for step=1,attOpt.seq_length do
	print(step)
	outputs[step] = unpack(model_attenLSTM:forward({step,inputs}))
--	outputs[step] = model:forward({step,inputs})
	print(criterion:forward(outputs[step], targets[step]))
end
---------------backward
gradOutputs, gradInputs = {}, {}
for step=attOpt.seq_length,1,-1 do
  gradOutputs[step] = criterion:backward(outputs[step], targets[step])
  gradInputs[step] = model_attenLSTM:backward({step,inputs}, gradOutputs[step])
end
--------------update
--model_attenLSTM:updateParameters(0.01)
model_attenLSTM:forget()
model_attenLSTM:zeroGradParameters()

--------------test
for step=1,attOpt.seq_length do
        print(step)
        outputs[step] = unpack(model_attenLSTM:forward({step,inputs}))
--      outputs[step] = model:forward({step,inputs})
        print(criterion:forward(outputs[step], targets[step]))
end
