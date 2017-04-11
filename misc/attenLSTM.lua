require 'nn'
require 'os'
local utils = require 'misc.utils'
local attenmodel = require 'misc.attenmodel'
local layer, parent = torch.class('nn.attenLSTM','nn.Module')

function layer:__init(opt) 
	parent.__init(self)
	self.feat_dim = utils.getopt(opt, 'feat_dim') 
    	self.hidden_size = utils.getopt(opt, 'hidden_size')
	self.seq_length = utils.getopt(opt, 'seq_length')
	self.class_num =  utils.getopt(opt, 'class_num')
	self.batch_size = utils.getopt(opt, 'batch_size')
	
	self.atten = attenmodel.subatten(self.seq_length, self.feat_dim, self.hidden_size, self.batch_size)
	zero = 1
	if zero == 1 then
		self.rnn = nn.LSTM(self.feat_dim, self.hidden_size):maskZero(1)
		self.softmax = nn.Sequential()
			:add(nn.MaskZero(nn.Linear(self.hidden_size, self.class_num),1))
			:add(nn.MaskZero(nn.LogSoftMax(),1))
	else
		self.rnn = nn.LSTM(self.feat_dim, self.hidden_size)
		self.softmax = nn.Sequential()
        	        :add(nn.Linear(self.hidden_size, self.class_num))
	                :add(nn.LogSoftMax())
	end
	self.atten_out = torch.Tensor(self.batch_size, self.feat_dim):fill(0)

end

function layer:getModulesList()
	return { self.atten, self.rnn, self.softmax}
end

function layer:parameters()
	local p1,g1 = self.atten:parameters()
	local p2,g2 = self.rnn:parameters()
	local p3,g3 = self.softmax:parameters()

	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end
	for k,v in pairs(p3) do table.insert(params, v) end
	
	local grad_params = {}
	for k,v in pairs(g1) do table.insert(grad_params, v) end
	for k,v in pairs(g2) do table.insert(grad_params, v) end
	for k,v in pairs(g3) do table.insert(grad_params, v) end

	return params, grad_params
end

function layer:training()
	self.atten:training()
	self.rnn:training()
	self.softmax:training()
end

function layer:evaluate()
	self.atten:evaluate()
	self.rnn:evaluate()
	self.softmax:evaluate()
end

function layer:updateOutput(input)
	local step = input[1]
	local feats = input[2]
	local featstmp = input[3]
	if step == 1 then
		self.prevHidden = torch.Tensor(self.batch_size,self.hidden_size):fill(0)
		self.prevCell = torch.Tensor(self.batch_size,self.hidden_size):fill(0)
	else 
		--[[
		local prevs = self.rnn:getHiddenState(step,self.atten_out)
	        self.prevHidden = prevs[1]
		print(self.prevHidden)
	        self.prevCell = prevs[2]
		--]]
		tmp = 1
	end
	self.atten_out = self.atten:forward({feats,self.prevHidden})
	--print(self.atten_out)

	self.lstm_out = self.rnn:forward(self.atten_out)
	--print(self.rnn:forward(featstmp))
	local prevs = self.rnn:getHiddenState(step,self.atten_out)
	self.prevHidden = prevs[1]
	self.prevCell = prevs[2]
	--print(self.lstm_out)

	self.class_out = self.softmax:forward(self.lstm_out)
	--print(self.class_out)
	return {self.lstm_out}
end

function layer:updateGradInput(input, gradOutput)
        local step = input[1]
        local feats = input[2]

	local d_softmax = self.softmax:backward(self.lstm_out, gradOutput)
	local d_attenout = self.rnn:backward(self.atten_out, d_softmax)
	local dummy = self.atten:backward({feats, self.prevHidden}, d_attenout)

	return {dummy}
end
