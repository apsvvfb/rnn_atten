require 'rnn'
require 'os'
require 'nn'
require 'optim'

printout=1
debug=0
rnn= nn.Sequential()
      :add(nn.FastLSTM(4,10))
      :add(nn.Linear(10,2))
      :add(nn.LogSoftMax())
----------------------don't change the order-----------------------
rnn2=rnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
-- get weights and loss wrt weights from the model
x, dl_dx = rnn:getParameters()
--------------------------------------------------------------------

inputs = {torch.randn(3,4), torch.randn(3,4), torch.randn(3,4)}
--targets= {torch.randn(3,1), torch.randn(3,1), torch.randn(3,1)}
targets = {}
table.insert(targets, torch.Tensor(3):fill(1))
table.insert(targets, torch.Tensor(3):fill(1))
table.insert(targets, torch.Tensor(3):fill(1))
nStep=3

------------------------------------------ single step
outputs, err = {}, 0
criterion = nn.ClassNLLCriterion()
for step=1,3 do
   outputs[step] = rnn:forward(inputs[step])
   if printout==1 and step==1 then
	print(outputs[step])
   end
   --err = err + criterion:forward(outputs[step], targets[step])
   criterion:forward(outputs[step], targets[step])
end

gradOutputs, gradInputs = {}, {}
for step=3,1,-1 do
  gradOutputs[step] = criterion:backward(outputs[step], targets[step])
  gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
end

x1=x:clone()
dl_dx1=dl_dx:clone()

rnn:updateParameters(0.01)
rnn:forget()
rnn:zeroGradParameters()

if printout==1 then
	print(rnn:forward(inputs[1]))
end

-------------------------------------- sequencer
seq = nn.Sequencer(rnn2)
out=seq:forward(inputs)
if printout==1 then
	print(out[1])
end
x, dl_dx = seq:getParameters()

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

outputs = seq:forward(inputs)
err = criterion:forward(outputs, targets)
gradOutputs = criterion:backward(outputs, targets)
gradInputs = seq:backward(inputs, gradOutputs)

if debug==1 then
	print(x-x1)
	print(dl_dx-dl_dx1)
end

seq:updateParameters(0.01)
seq:zeroGradParameters()

out=seq:forward(inputs)
if printout==1 then
print(out[1])
end
