local AddTensor, parent = torch.class('nn.AddTensor', 'nn.Module')

function AddTensor:__init(tensor_add)
   parent.__init(self)
  
   self.tensor_add = tensor_add

end

function AddTensor:updateOutput(input)
	self.output:resizeAs(input):copy(input)
	if input:isSameSizeAs(self.tensor_add) then
		self.output:add(self.tensor_add)
	else
		print("error")
	end

   return self.output
end

function AddTensor:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput) 
      return self.gradInput
   end
end

function AddTensor:accGradParameters(input, gradOutput, scale)
      scale = scale or 1
      if input:isSameSizeAs(self.tensor_add) then
         self.tensor_add:add(scale, gradOutput)
      else
         print("accGradParameters:error")
      end
end
