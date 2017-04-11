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
params, gradParams = rnn:getParameters()
--------------------------------------------------------------------

sgd_params = {
   learningRate = 0.01,
   --learningRateDecay = 1e-4,
   --weightDecay = 0.005,
   --momentum = 0.9
}

inputs = {torch.randn(3,4), torch.randn(3,4), torch.randn(3,4)}
--targets= {torch.randn(3,1), torch.randn(3,1), torch.randn(3,1)}
targets = {}
table.insert(targets, torch.Tensor(3):fill(1))
table.insert(targets, torch.Tensor(3):fill(1))
table.insert(targets, torch.Tensor(3):fill(1))
nStep=3

------------------------------------------ single step
outputs, loss = {}, 0
criterion = nn.ClassNLLCriterion()
for step=1,3 do
   outputs[step] = rnn:forward(inputs[step])
   if printout==1 and step == 1 then
	print(outputs[step])
   end
   loss = loss + criterion:forward(outputs[step], targets[step])
   --criterion:forward(outputs[step], targets[step])
end

gradOutputs, gradInputs = {}, {}
rnn:zeroGradParameters()
for step=3,1,-1 do
  gradOutputs[step] = criterion:backward(outputs[step], targets[step])
  gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
end
--gradParams=gradInputs[1]:copy()

--rnn:updateParameters(0.01)
--rnn:forget()
--rnn:zeroGradParameters()

feval = function(params_new)
        -- copy the weight if are changed
        if params ~= params_new then
                params:copy(params_new)
        end
        -- select a training batch
        --local inputs, targets = nextBatch()

        -- reset gradients (gradients are always accumulated, to accommodate
        -- batch methods)
        --dl_dx:zero()

        -- evaluate the loss function and its derivative with respect to x, given a mini batch

        --local outputs = seq:forward(inputs)
        --local loss_x = criterion:forward(outputs, targets)
        --local gradOutputs = criterion:backward(outputs, targets)
        --local gradInputs = seq:backward(inputs, gradOutputs)
        return loss, gradParams
end
gradParams1=gradParams:clone()
params1=params:clone()
_, fs = optim.sgd(feval,params, sgd_params)
rnn:forget()
rnn:zeroGradParameters()
if printout==1 then
	print(rnn:forward(inputs[1]))
end
-------------------------------------- sequencer
seq = nn.Sequencer(rnn2)
if printout==1 then
	out=seq:forward(inputs)
	print(out[1])
end
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

sgd_params = {
   learningRate = 0.01,
   --learningRateDecay = 1e-4,
   --weightDecay = 0.005,
   --momentum = 0.9
}

-- get weights and loss wrt weights from the model
params, gradParams = seq:getParameters()
-- In the following code, we define a closure, feval, which computes
-- the value of the loss function at a given point x, and the gradient of
-- that function with respect to x. weigths is the vector of trainable weights,
-- it extracts a mini_batch via the nextBatch method
feval = function(params_new)
	-- copy the weight if are changed
	if params ~= params_new then
		params:copy(params_new)
	end
	-- select a training batch
	--local inputs, targets = nextBatch()

	-- reset gradients (gradients are always accumulated, to accommodate
	-- batch methods)
	gradParams:zero()

	-- evaluate the loss function and its derivative with respect to x, given a mini batch
	
	local outputs = seq:forward(inputs)
	local loss = criterion:forward(outputs, targets)
	local gradOutputs = criterion:backward(outputs, targets)
	local gradInputs = seq:backward(inputs, gradOutputs)
	if debug==1 then
		print(gradParams-gradParams1)
	end
	--print(params-params1)
	return loss, gradParams
end

--seq:updateParameters(0.01)
--seq:zeroGradParameters()

_, fs = optim.sgd(feval,params, sgd_params)

if printout==1 then
	out=seq:forward(inputs)
	print(out[1])
end
