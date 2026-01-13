import torch

"""
    https://github.com/binarycopycode/CRP_DAM
    Modified for rBRP 
"""
class Env():
    def __init__(self, device, x):
        super().__init__()
        #x: (batch_size) X (max_stacks) X (max_tiers)
        self.device = device
        self.x = x
        self.batch, self.max_stacks,self.max_tiers=x.size()
        self.target_stack = None
        self.empty = torch.zeros([self.batch], dtype=torch.bool).to(self.device)
        self.retrieved = torch.ones([self.batch], dtype=torch.bool).to(self.device)
        self.prev_action = torch.zeros([self.batch, 2]).to(self.device)
        self.retrieved = torch.zeros([self.batch])
        #True -> Empty / False-> not Empty
    def find_target_stack(self):
        mx_val = torch.max(self.x, dim=2)[0].to(self.device)
        self.target_stack = torch.argmax(mx_val, dim=1).to(self.device)
    def _update_empty(self):
        bottom_val = self.x[:,:,0].to(self.device) 
        batch_mx = torch.max(bottom_val, dim=1)[0].to(self.device) #Max 
        self.empty = torch.where(batch_mx>0., False, True).to(self.device) #if batch_mx is 0 => Empty
    def clear(self):
        #Retrieval Process
        self.find_target_stack()
        binary_x = torch.where(self.x > 0., 1, 0).to(self.device) # Block -> 1 Empty -> 0
        stack_len = torch.sum(binary_x, dim=2).to(self.device) #Length of stack
        block_nums = torch.sum(stack_len, dim=1).to(self.device)
        target_stack_len = torch.gather(stack_len, dim=1, index = self.target_stack[:,None].to(self.device)) #target_stack의 location
        stack_mx_index = torch.argmax(self.x, dim=2).to(self.device)
        target_stack_mx_index = torch.gather(stack_mx_index, dim=1, index=self.target_stack[:,None].to(self.device))
        clear_mask = ((target_stack_len -1) == target_stack_mx_index)
        clear_mask = clear_mask
        clear_mask = clear_mask & (torch.where(target_stack_len > 0, True, False))
        self.retrieved = clear_mask.squeeze(-1)
        while torch.sum(clear_mask.squeeze(-1))>0: #Until there are no available retrievals
            batch_mask = clear_mask.repeat_interleave(self.max_stacks * self.max_tiers).to(self.device)
            batch_mask = torch.reshape(batch_mask, (self.batch, self.max_stacks, self.max_tiers)).to(self.device)

            mask = torch.zeros((self.batch, self.max_stacks, self.max_tiers), dtype=torch.bool).to(self.device)
            input_index = (torch.arange(self.batch).to(self.device), self.target_stack, target_stack_len.squeeze(-1).to(self.device) - 1)
            mask = mask.index_put(input_index, torch.tensor(True).to(self.device)).to(self.device)
            
            mask = mask & batch_mask
            mask = mask.to(self.device)
            self.x = self.x.masked_fill((mask == True).to(self.device), 0.)

            self.find_target_stack()
            len_mask = torch.where(self.x > 0., 1, 0).to(self.device)
            stack_len = torch.sum(len_mask, dim=2).to(self.device)
            target_stack_len = torch.gather(stack_len, dim=1, index=self.target_stack[:, None].to(self.device)).to(self.device)
            stack_mx_index = torch.argmax(self.x, dim=2).to(self.device)
            target_stack_mx_index = torch.gather(stack_mx_index, dim=1, index=self.target_stack[:, None].to(self.device)).to(self.device)
            clear_mask = ((target_stack_len - 1) == target_stack_mx_index)
            clear_mask = clear_mask.to(self.device)
            clear_mask = clear_mask & (torch.where(target_stack_len > 0, True, False).to(self.device))
        
        binary_x = torch.where(self.x > 0., 1, 0).to(self.device) 
        stack_len = torch.sum(binary_x, dim=2).to(self.device) 
        new_block_nums = torch.sum(stack_len, dim=1).to(self.device)
        new_ratio = (block_nums + 1)/(new_block_nums+1)
        #Priority Update (to 1)
        self.last_retrieved_nums = block_nums - new_block_nums
        self.x = torch.mul(self.x.view(self.batch, self.max_stacks*self.max_tiers), new_ratio.unsqueeze(1)).view(self.batch, self.max_stacks, self.max_tiers).to(self.device)
        
        self._update_empty()
    def all_empty(self):
        """ Return if all states are at terminal.
        """
        sum = torch.sum(self.empty.type(torch.int))
        if (sum == self.batch):
            return True
        else:
            return False
    def step_rBRP(self, next_node):
        """ next_node : (batch, 1) int, range[0, max_stacks)

            mask(batch,max_stacks,1) 
            context: (batch, 1, embed_dim)
        """

        len_mask = torch.where(self.x > 0., 1, 0).to(self.device)
        stack_len = torch.sum(len_mask, dim=2)

        target_stack_len = torch.gather(stack_len, dim=1, index=self.target_stack[:, None]).to(self.device)

        next_stack_len=torch.gather(stack_len,dim=1,index=next_node).to(self.device)
        top_ind=stack_len-1
        top_ind=torch.where(top_ind>=0,top_ind,0).to(self.device)
        top_val=torch.gather(self.x,dim=2,index=top_ind[:,:,None]).to(self.device)
        top_val=top_val.squeeze(-1)
        target_top_val=torch.gather(top_val,dim=1,index=self.target_stack[:,None]).to(self.device)
        target_ind=target_stack_len-1
        target_ind=torch.where(target_ind>=0,target_ind,0).to(self.device)
        input_index=(torch.arange(self.batch).to(self.device),self.target_stack.to(self.device),target_ind.squeeze(-1).to(self.device))
        self.x=self.x.index_put_(input_index,torch.Tensor([0.]).to(self.device))

        input_index=(torch.arange(self.batch).to(self.device),next_node.squeeze(-1).to(self.device),next_stack_len.squeeze(-1).to(self.device))
        self.x=self.x.index_put_(input_index,target_top_val.squeeze(-1)).to(self.device)
        self.clear()


    def step_uBRP(self, actions):
        """ action: (batch, 2) int, range[0,max_stacks)
            It is constructed with (Source, Destination)
            no invalid action (Assume)
        """
        len_mask = torch.where(self.x > 0., 1, 0).to(self.device)
        stack_len = torch.sum(len_mask, dim=2)
        source_index = actions[:,0]
        dest_index = actions[:,1]
        source_stack_len = torch.gather(stack_len, dim=1, index=source_index[:,None]).to(self.device)
        dest_stack_len = torch.gather(stack_len, dim=1, index=dest_index[:,None]).to(self.device)
        top_ind = stack_len - 1
        top_ind = torch.where(top_ind >=0, top_ind, 0).to(self.device)
        top_val = torch.gather(self.x, dim=2, index=top_ind[:,:,None]).to(self.device)
        top_val = top_val.squeeze(-1)
        source_top_val = torch.gather(top_val, dim=1, index=source_index[:,None]).to(self.device)
        source_ind = source_stack_len - 1
        source_ind = torch.where(source_ind >=0, source_ind, 0).to(self.device)
        input_index = (torch.arange(self.batch).to(self.device), source_index.to(self.device), source_ind.squeeze(-1).to(self.device))
        self.x = self.x.index_put_(input_index, torch.Tensor([0.]).to(self.device))
        input_index = (torch.arange(self.batch).to(self.device), dest_index.to(self.device), dest_stack_len.squeeze(-1).to(self.device))
        self.x = self.x.index_put_(input_index, source_top_val.squeeze(-1)).to(self.device)
        self.clear()
        self.prev_action = actions


    def create_mask_rBRP(self):
        top_val=self.x[:,:,-1]
        mask=torch.where(top_val>0,True,False).to(self.device)
        mask = mask.bool()
        a=self.target_stack.clone().to(self.device)
        index = (torch.arange(self.batch).to(self.device), a.squeeze())
        mask=mask.index_put(index,torch.BoolTensor([True] ).to(self.device))
        return mask[:,:,None].to(self.device)
    
    def create_mask_uBRP(self):
        top_val=self.x[:,:,-1].to(self.device)
        bottom_val = self.x[:,:,0].to(self.device)
        #Mask 1. Dest stack is full
        mask_top=torch.where(top_val>0,True,False).to(self.device).bool()
        mask_top = mask_top.repeat(1, self.max_stacks).view(self.batch, self.max_stacks, self.max_stacks).to(self.device)
        #Mask 2. Source stack is empty
        mask_bottom=torch.where(bottom_val==0.,True,False).to(self.device).bool()
        mask_bottom = mask_bottom.view(self.batch, self.max_stacks, 1).repeat(1, 1, self.max_stacks).to(self.device)

        #Mask 3. Dest == Source
        mask_diagonal = torch.zeros((self.batch, self.max_stacks, 2), dtype=torch.long).to(self.device)
        mask_diagonal[:,:,0] = torch.arange(self.max_stacks).to(self.device)
        mask_diagonal[:,:,1] = torch.arange(self.max_stacks).to(self.device)

        #Combine Masks
        mask = torch.logical_or(mask_top, mask_bottom).to(self.device)
        mask.scatter_(2, mask_diagonal, 1)
        return mask.view(self.batch, self.max_stacks*self.max_stacks)[:,:,None].to(self.device)