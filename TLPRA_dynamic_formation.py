import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.autograd.functional as AGF
from pytorch_lightning.callbacks import EarlyStopping
import math
import torch.linalg as linalg
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
from numpy import cos, sin, arccos, arctan2, sqrt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch.optim.lr_scheduler import StepLR
from datetime import datetime 

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size":20,
    "font.sans-serif": ["Helvetica"]})

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
device = torch.device("cuda")
alp=1.0
damp_m=0.001

class Srelu(nn.Module):#定义Srelu
    def __init__(self) :
        super().__init__()

    def forward(self, z):
        z0_ = z.clone()
        d = torch.tensor(.01)
        z0_[(z0_>= d)] = z[(z0_>= d)]
        z0_[(z0_ <= 0.0)] = 0.0
        z0_[torch.logical_and(z0_ < d, z0_ > 0.0)] = z[torch.logical_and(z < d, z > 0.0)]**2/(2*d)
        return z0_


class ICNN(nn.Module):  # represents the controller gain
    def __init__(self):
        super(ICNN, self).__init__()
        self.w_z0=nn.Parameter(torch.Tensor(2,32).uniform_(-0.1,0.1))
        self.w_y1=nn.Parameter(torch.Tensor(2,32).uniform_(-0.1,0.1))
        self.w_y2=nn.Parameter(torch.Tensor(2,32).uniform_(-0.1,0.1))
        # self.w_y3=nn.Parameter(torch.Tensor(2,32).uniform_(-0.1,0.1))
        self.w_z1=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_z2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        # self.w_z3=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_yout=nn.Parameter(torch.Tensor(2,1).uniform_(-0.1,0.1))
        self.w_zout=nn.Parameter(torch.Tensor(32,1).uniform_(-0.1,0.1))

        self.srleu=Srelu()
        
    def forward(self, z):
        
        z0 = z.clone()
        z1 = z0 @ self.w_z0 
        z1s =self.srleu(z1)
        z2 = z1s @ torch.relu(self.w_z1)   +z0 @ self.w_y1 
        z2s = self.srleu(z2)
        z3 =  z2s @ torch.relu(self.w_z2)  + z0 @ self.w_y2
        z3s = self.srleu(z3)
        # z4 =  z3s @ torch.relu(self.w_z3)  + z0 @ self.w_y3
        # z4s = self.srleu(z4)
        zout =  z3s @ torch.relu(self.w_zout)  + z0 @ self.w_yout
        zouts = self.srleu(zout)

        return zouts


class Damping(nn.Module):  # represents the controller gain
    def __init__(self):
        super(Damping, self).__init__()
        N = 2
        self.offdiag_output_dim = N*(N-1)//2
        self.diag_output_dim = N
        self.output_dim = self.offdiag_output_dim + self.diag_output_dim
        damp_min=torch.tensor([damp_m,damp_m])
        self.damp_min = damp_min

        self.w_d1=nn.Parameter(torch.Tensor(2,32).uniform_(-0.1,0.2))
        self.w_d2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.2))
        self.w_d3=nn.Parameter(torch.Tensor(32,2).uniform_(-0.1,0.2))
        self.w_o1=nn.Parameter(torch.Tensor(2,32).uniform_(-0.1,0.2))
        self.w_o2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.2))
        self.w_o3=nn.Parameter(torch.Tensor(32,1).uniform_(-0.1,0.2))

        self.b_d1=nn.Parameter(torch.zeros(32))
        self.b_d2=nn.Parameter(torch.zeros(32))
        self.b_d3=nn.Parameter(torch.zeros(2))
        self.b_o1=nn.Parameter(torch.zeros(32))
        self.b_o2=nn.Parameter(torch.zeros(32))
        self.b_o3=nn.Parameter(torch.zeros(1))

    def forward(self, input):

        x0 = input
        x=x0.clone()
        z=x0.clone()

        d1 = x @ self.w_d1   + self.b_d1
        d1t = torch.tanh(d1)
        d2 =  d1t @ self.w_d2 + self.b_d2
        d2t = torch.tanh(d2)
        d3 = d2t @ self.w_d3  + self.b_d3
        d3r = (torch.relu(d3)+self.damp_min) * x

        n = self.diag_output_dim
        diag_idx = np.diag_indices(n)
        off_diag_idx = np.tril_indices(n, k=-1)
        D = torch.zeros(x.shape[0], n)

        o1 =  x @ self.w_o1 + self.b_o1
        o1t = torch.tanh(o1)
        o2 = o1t @ self.w_o2  + self.b_o2
        o2t = torch.tanh(o2)
        o3 = o2t @ self.w_o3  + self.b_o3


        L = torch.zeros(n, n)
        diag_elements = d3r
        off_diag_elements = o3
        L[off_diag_idx] = off_diag_elements
        L[diag_idx] = diag_elements
        D_temp = L@L.t()
        D = D_temp @ x
        return D

# class RBFNN(nn.Module):
#     def __init__(self) -> None:
#         super(RBFNN,self).__init__()
#         self.input_dim = input_dim=8
#         self.hidden_dim = hidden_dim=32
#         self.output_dim = output_dim=2
#         self.num_centers = num_centers=2048
#         self.dt=dt=0.01
        
#         # RBF层的参数
#         self.centers = torch.Tensor(num_centers, input_dim).normal_(-3,3)
#         self.widths = torch.ones(num_centers)*3
        
#         # 全连接层
#         self.w = torch.rand(num_centers,output_dim)
#         self.mu=torch.tensor([1])
        

#     def forward(self, q,dq,e,E):
#         # 计算RBF层的输出
#         x=torch.cat((q.clone(),dq.clone(),e.clone(),E.clone())).unsqueeze(0)
#         rbf = torch.exp(-torch.sum((x.unsqueeze(1) - self.centers)**2, dim=2) / (2*self.widths**2))
        
#         # 计算全连接层的输出
#         output = rbf@self.w
#         return output
    
#     def update_RBFNN(self,q,dq,phi,dphi,E):
#         x=torch.cat((q.clone(),dq.clone(),phi.clone(),dphi.clone())).unsqueeze(0)
#         rbf = torch.exp(-torch.sum((x.unsqueeze(1) - self.centers)**2, dim=2) / (2*self.widths**2))
#         dotW=-100*(rbf.t()@E.clone().unsqueeze(0)+self.mu*self.w)
#         dotmu=-0.5*self.mu
#         self.w=self.w+self.dt*dotW
#         self.mu=self.mu+self.dt*dotmu
        
        

class leaderfollower_NBS_tracking_control_learner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.delta=torch.tensor([[2,-1,0,-1],[-1,2,-1,0],[0,-1,2,-1],[-1,0,-1,2]])

        self.L=torch.tensor([[0,0,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        self.B=torch.tensor([1,0,0,0])
        self.B_d=torch.diag(self.B)

        # self.H=self.delta+self.B_d
        self.agents=4
        coord_dim=2
        self._coord_dim = coord_dim
        self._state_dim = coord_dim*2
        self._action_dim = coord_dim
        self.init_quad_pot = 1.0,
        self.min_quad_pot = 1e-3,
        self.max_quad_pot = 1e1,
        self.icnn_min_lr = 1e-1,
        self.alpha=torch.eye(2)*torch.tensor([0])

        self.S = torch.eye(coord_dim)*torch.Tensor([1.0])

        self.icnn_module=ICNN()
        self.damping_module=Damping()
        self.icnn_module_list=[]
        self.damping_module_list=[]
        self.rbf_module_list=[]
        for i in range(self.agents):
            # self.rbf_module_list.append(RBFNN())
            self.damping_module_list.append(Damping())
            self.icnn_module_list.append(ICNN())
        self.T=1000
        self.state = torch.zeros((self.T+1,self._state_dim*self.agents),requires_grad=True).to(self.device)
        self.state_error = torch.zeros((self.T,self._state_dim*self.agents),requires_grad=True).to(self.device)
        self.state_d = torch.zeros((self.T,self._coord_dim*3)).to(self.device)
        self.control = torch.zeros((self.T,self._action_dim*self.agents),requires_grad=True).to(self.device)
        self.control_wg = torch.zeros((self.T,self._action_dim*self.agents),requires_grad=True).to(self.device)
        self.stage_cost = torch.zeros((self.T+1),requires_grad=True).to(self.device)

        self.e=torch.zeros((self.T,self.agents,self._coord_dim),requires_grad=True).to(self.device)
        self.e_dot=torch.zeros((self.T,self.agents,self._coord_dim),requires_grad=True).to(self.device)
        self.E=torch.zeros((self.T,self.agents,self._coord_dim),requires_grad=True).to(self.device)
        self.dphi_de=torch.zeros((self.T,self.agents,self._action_dim),requires_grad=True).to(self.device)
        self.ddphi_dde=torch.zeros((self.T,self.agents,self._action_dim,self._action_dim),requires_grad=True).to(self.device)
        self.phi_=torch.zeros((self.T,self.agents,self._coord_dim),requires_grad=True).to(self.device)
        self.dphi_dt=torch.zeros((self.T,self.agents,self._coord_dim),requires_grad=True).to(self.device)
        self.u_phi=torch.zeros((self.T,self.agents,self._action_dim),requires_grad=True).to(self.device)
        
        self.form_state=torch.zeros((self.T,self.agents,self._coord_dim*3),requires_grad=True).to(self.device)

        self.dt=torch.tensor(0.01)
        self.m1=torch.tensor(1.0)
        self.m2=torch.tensor(1.0)
        self.l1=torch.tensor(1.0)
        self.l2=torch.tensor(1.0)
        self.g = torch.tensor(9.8)
        self.flag = 0
        
    def set_alpha(self, a=0):
        self.alpha=torch.eye(2)*torch.tensor([a])
        return self.alpha

    def phi(self, x,i):
        y=self.icnn_module_list[i](x)+  (x @ self.S @ x.t())
        return y


    def get_action(self, e,E_bar):
        state = torch.zeros((1,4), requires_grad=False).to(self.device)
        z1 = e.requires_grad_() 
      
        psi = self.phi(z1)

        self.u_pot_1 = torch.autograd.grad(psi,z1,create_graph=True,retain_graph=True)
        self.u_pot = self.u_pot_1[0]
        z2=E_bar+self.u_pot_1[0]

        self.u_dpot_1 = AGF.hessian(self.phi,z1,create_graph=True)
        self.u_dpot = self.u_dpot_1.clone().squeeze()
        self.u_damp = self.damping_module(z2)
       
        return self.u_pot,self.u_dpot, self.u_damp 
    
    def Gravity(self, x):
        
        b_2=x[1].clone()
        M=torch.tensor([[(self.m1+self.m2)*self.l1**2+self.m2*self.l2**2+2*self.m2*self.l1*self.l2*torch.cos(b_2),self.m2*self.l2**2+self.m2*self.l2*self.l1*torch.cos(b_2)],
                                    [self.m2*self.l1*self.l2*torch.cos(b_2)+self.m2*self.l2**2,self.m2*self.l2**2]])
        return M
    
    def Correlation(self,state):
        b_1=state[0].clone()
        b_2=state[1].clone()
        v_1=state[2].clone()
        v_2=state[3].clone()
        C=torch.tensor([[-self.m2*self.l1*self.l2*v_2*torch.sin(b_2),-self.m2*self.l1*self.l2*(v_1+v_2)*torch.sin(b_2)],
                        [self.m2*self.l1*self.l2*v_1*torch.sin(b_2),0]])
        return C

    def gravity_compensate(self,x):
        b1=x[0].clone()
        b2=x[1].clone()
        gc=torch.tensor([torch.tensor(2)*self.g*torch.cos(b1)+self.g*torch.cos(b1+b2),self.g*torch.cos(b1+b2)])

        return gc
    
    def f(self,state,control):
        x=state.clone()
        u=control.clone()
        dis=torch.tensor([1,1])
        # dis=2*torch.rand(1,2)-1
        b_1=x[0].clone()
        b_2=x[1].clone()
        v_1=x[2].clone()
        v_2=x[3].clone()
        u=u#+dis
        u=u.unsqueeze(0)
        M=self.Gravity(x)
        V=torch.tensor([[-self.m2*self.l1*self.l2*(2*v_1*v_2+v_2**2)*torch.sin(b_2)],[self.m2*self.l1*self.l2*(v_1**2)*torch.sin(b_2)]])
        G=torch.tensor([[(self.m1+self.m2)*self.g*self.l1*torch.cos(b_1)+self.m2*self.g*self.l2*torch.cos(b_1+b_2)],[self.m2*self.g*self.l2*torch.cos(b_1+b_2)]])
        ux=u.t()
        dot=torch.inverse(M) @ (ux-V-G)
        dx=torch.cat((x[2:],dot.t().squeeze()),0)    
   
        state_t=x+self.dt*dx
        # state_t.retain_grad()
        return state_t
    
    def f_dis(self,state,control):
        x=state.clone()
        u=control.clone()
        dis=torch.tensor([1,1])
        # dis=2*torch.rand(1,2)-1
        b_1=x[0].clone()
        b_2=x[1].clone()
        v_1=x[2].clone()
        v_2=x[3].clone()
        u=u+dis
        u=u.unsqueeze(0)
        M=self.Gravity(x)
        V=torch.tensor([[-self.m2*self.l1*self.l2*(2*v_1*v_2+v_2**2)*torch.sin(b_2)],[self.m2*self.l1*self.l2*(v_1**2)*torch.sin(b_2)]])
        G=torch.tensor([[(self.m1+self.m2)*self.g*self.l1*torch.cos(b_1)+self.m2*self.g*self.l2*torch.cos(b_1+b_2)],[self.m2*self.g*self.l2*torch.cos(b_1+b_2)]])
        ux=u.t()
        dot=torch.inverse(M) @ (ux-V-G)
        dx=torch.cat((x[2:],dot.t().squeeze()),0)    
   
        state_t=x+self.dt*dx
        # state_t.retain_grad()
        return state_t

    def forward(self, x):
        return 0

    def configure_optimizers(self):
        params_to_optimize = []
        for network in self.icnn_module_list:
            params_to_optimize += list(network.parameters())
        for network in self.damping_module_list:
            params_to_optimize += list(network.parameters())
        # optimizer = torch.optim.Adam([{'params':self.icnn_module_list[0].parameters()},{'params':self.damping_module_list[0].parameters()},
        #                               {'params':self.icnn_module_list[1].parameters()},{'params':self.damping_module_list[1].parameters()},
        #                               {'params':self.icnn_module_list[2].parameters()},{'params':self.damping_module_list[2].parameters()},
        #                               {'params':self.icnn_module_list[3].parameters()},{'params':self.damping_module_list[3].parameters()}], lr=1e-3)
        optimizer = torch.optim.Adam(params_to_optimize, lr=1e-3)
        scheduler =StepLR(optimizer,step_size=50, gamma=0.5)
        return [optimizer],[scheduler]


    def training_step(self, train_batch, batch_idx):
        # torch.autograd.set_detect_anomaly(True)
        loss = torch.zeros((1)).to(self.device)
        state = (self.state*torch.zeros(1)).to(self.device)
        state_d =(self.state_d*torch.zeros(1)).to(self.device)
        control =(self.control*torch.zeros(1)).to(self.device)
        stage_cost =(self.stage_cost* torch.zeros(1)).to(self.device)
        form_state=(self.form_state*torch.zeros(1)).to(self.device)
        state[0] = train_batch[0]
        e=(self.e* torch.zeros(1)).to(self.device)
        e_dot=(self.e_dot* torch.zeros(1)).to(self.device)
        E=(self.E* torch.zeros(1)).to(self.device)
        dphi_de=(self.dphi_de* torch.zeros(1)).to(self.device)
        ddphi_dde=(self.ddphi_dde* torch.zeros(1)).to(self.device)
        phi=(self.phi_* torch.zeros(1)).to(self.device)
        dphi_dt=(self.dphi_dt* torch.zeros(1)).to(self.device)
        u_phi=(self.u_phi*torch.zeros(1)).to(self.device)

        for t in range(self.T):
            # state_d[t] = torch.Tensor([sin(0.1*self.dt*t),cos(0.1*self.dt*t),0.1*cos(0.1*self.dt*t),-0.1*sin(0.1*self.dt*t),-0.1*0.1*sin(0.1*self.dt*t),-0.1*0.1*cos(0.1*self.dt*t)])
            state_d[t] = torch.Tensor([1,1,0,0,0,0])
            form_state[t]= torch.Tensor([[sin(0.1*self.dt*t+0*np.pi/2),cos(0.1*self.dt*t+0*np.pi/2),0.1*cos(0.1*self.dt*t+0*np.pi/2),-0.1*sin(0.1*self.dt*t+0*np.pi/2),-0.1*0.1*sin(0.1*self.dt*t+0*np.pi/2),-0.1*0.1*cos(0.1*self.dt*t+0*np.pi/2)],
                                    [sin(0.1*self.dt*t+1*np.pi/2),cos(0.1*self.dt*t+1*np.pi/2),0.1*cos(0.1*self.dt*t+1*np.pi/2),-0.1*sin(0.1*self.dt*t+1*np.pi/2),-0.1*0.1*sin(0.1*self.dt*t+1*np.pi/2),-0.1*0.1*cos(0.1*self.dt*t+1*np.pi/2)],
                                    [sin(0.1*self.dt*t+2*np.pi/2),cos(0.1*self.dt*t+2*np.pi/2),0.1*cos(0.1*self.dt*t+2*np.pi/2),-0.1*sin(0.1*self.dt*t+2*np.pi/2),-0.1*0.1*sin(0.1*self.dt*t+2*np.pi/2),-0.1*0.1*cos(0.1*self.dt*t+2*np.pi/2)],
                                    [sin(0.1*self.dt*t+3*np.pi/2),cos(0.1*self.dt*t+3*np.pi/2),0.1*cos(0.1*self.dt*t+3*np.pi/2),-0.1*sin(0.1*self.dt*t+3*np.pi/2),-0.1*0.1*sin(0.1*self.dt*t+3*np.pi/2),-0.1*0.1*cos(0.1*self.dt*t+3*np.pi/2)]])

            q_d = state_d[t,0:2]
            dq_d=state_d[t,2:4]
            ddq_d=state_d[t,4:6]
            for i in range(self.agents):
                state_i=state[t,self._state_dim*i:self._state_dim*(i+1)]
                q_i=state_i[:self._coord_dim]
                dq_i=state_i[self._coord_dim:]
                form_q_i=form_state[t,i,:self._coord_dim]
                form_dq_i=form_state[t,i,self._coord_dim:self._state_dim]
                form_ddq_i=form_state[t,i,self._state_dim:]

                for j in range(self.agents):
                    state_j=state[t,self._state_dim*j:self._state_dim*(j+1)]
                    q_j=state_j[:self._coord_dim]
                    dq_j=state_j[self._coord_dim:]
                    form_q_j=form_state[t,j,:self._coord_dim]
                    form_dq_j=form_state[t,j,self._coord_dim:self._state_dim]
                    form_ddq_j=form_state[t,j,self._state_dim:] 


                    e[t,i]=e[t,i]+self.L[i,j]*(q_i-form_q_i-(q_j-form_q_j))
                    e_dot[t,i]=e_dot[t,i]+self.L[i,j]*(dq_i-form_dq_i-(dq_j-form_dq_j))
                e[t,i]=e[t,i]+self.B[i]*(q_i-form_q_i-q_d)
                e_dot[t,i]=e_dot[t,i]+self.B[i]*(dq_i-form_dq_i-dq_d)

                z1 = e[t,i].requires_grad_() 
                dphi_de[t,i]=AGF.jacobian(lambda x: self.phi(x, i),z1,create_graph=True).squeeze()
                phi[t,i]=form_dq_i-dphi_de[t,i]
                E[t,i]=dq_i-phi[t,i]
                ddphi_dde[t,i]=AGF.hessian(lambda x: self.phi(x, i),z1,create_graph=True).squeeze()
                dphi_dt[t,i]=(form_ddq_i-e_dot[t,i].clone()@ ddphi_dde[t,i].clone())
            
            for i in range(self.agents):
                state_i=state[t,self._state_dim*i:self._state_dim*(i+1)]
                q_i=state_i[:self._coord_dim]
                dq_i=state_i[self._coord_dim:]
                # u_phi=torch.zeros(self._action_dim)
                for j in range(self.agents):
                    u_phi[t,i]=u_phi[t,i]+self.L[i,j]*(dphi_de[t,i]-dphi_de[t,j])
                u_phi[t,i]=u_phi[t,i]+self.B[i]*dphi_de[t,i]
                gc=self.gravity_compensate(state_i)
                C=self.Correlation(state_i)
                M=self.Gravity(state_i)
                control[t,self._action_dim*i:self._action_dim*(i+1)] = gc.clone() +dphi_dt[t,i]@ M.clone().t() + phi[t,i] @ C.clone().t()-self.damping_module_list[i](E[t,i])-u_phi[t,i] 
                
                state[t+1,self._state_dim*i:self._state_dim*(i+1)] = self.f(state_i,control[t,self._action_dim*i:self._action_dim*(i+1)])            
            stage_cost[t] =(e[t,0]).clone()@(e[t,0]).clone().t()  +(e[t,1]).clone()@(e[t,1]).clone().t() +(e[t,2]).clone()@(e[t,2]).clone().t() +(e[t,3]).clone()@(e[t,3]).clone().t() 
        z0=torch.Tensor([0,0]).requires_grad_()
        regularizer=torch.zeros(1)
        for i in range(self.agents):   
            ddPhi_ddz0= AGF.hessian(lambda x: self.phi(x, i),z0,create_graph=True).squeeze()
            A=self.alpha-ddPhi_ddz0
            reg_value,reg_vector=torch.linalg.eig(A)
            real_reg=reg_value.real
            regularizer=regularizer+torch.relu(real_reg.max()).sum()
        
        loss = loss + 10*torch.sum(stage_cost)+10*regularizer
        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss = torch.zeros((1)).to(self.device)
        T=5000
        state = torch.zeros((T+1,4)).to(self.device)
        state_error = torch.zeros((T,4)).to(self.device)
        state_d = torch.zeros((T,4)).to(self.device)
        control = torch.zeros((T,2)).to(self.device)
        stage_cost = torch.zeros((T+1)).to(self.device)
        state[0,0:4] = test_batch
        for i in range(T):
            t=i
            state_d[i] = torch.Tensor([sin(0.1*self.dt*t),cos(0.1*self.dt*t),0.1*cos(0.1*self.dt*t),-0.1*sin(0.1*self.dt*t),-0.1*0.1*sin(0.1*self.dt*t),-0.1*0.1*cos(0.1*self.dt*t)])            # state_d[t] = torch.Tensor([1,1,0,0,0,0])
            q_d = state_d[i,0:2]
            dq_d=state_d[i,2:4]
            ddq_d=state_d[i,4:6]
            q=state[i,0:2]
            dq=state[i,2:4]
            self.get_action(state[i]-state_d[i,0:4])
            gc=self.gravity_compensate(state[i])
            C=self.Correlation(state[i])
            M=self.Gravity(state[i])
            control[i] = gc +(ddq_d-(dq-dq_d) @ self.u_dpot) @ M.t() + (dq_d-self.u_pot) @ C.t()-self.u_pot-self.u_damp 
            state[i+1] = self.f(state[i],control[i])
            state_error[i]=state[i+1]-state_d[i,:4]
            stage_cost[i] = (state_error[i]).clone() @ (state_error[i]).clone().t() 
        z0=torch.zeros([0,0])   
        ddPhi_ddz0 = AGF.hessian(self.phi,z0,create_graph=True)
        ddPhi_ddz0= ddPhi_ddz0.clone().squeeze()
        A=self.alpha-ddPhi_ddz0.t() @ ddPhi_ddz0
        reg_value,reg_vector=torch.linalg.eig(A)
        real_reg=reg_value.real
        loss = loss + torch.sum(stage_cost)+torch.relu(real_reg.max()).sum()
        self.log('test_loss', loss)
    

    def plottrajetory(self,i_num,a_print=0):
        T=20000#sim time
        i_num=i_num
        state = torch.zeros((T+1,self._state_dim*self.agents),requires_grad=False).to(self.device)
        state_d = torch.zeros((T,self._coord_dim*3)).to(self.device)
        control = torch.zeros((T,self._action_dim*self.agents),requires_grad=False).to(self.device)
        form_state = torch.zeros((T,self.agents,self._coord_dim*3)).to(self.device)
        error=[]
        if i_num==1:
            state[0] = torch.tensor([0,0,0,0,1,1,0,0,-1,1,0,0,1,-1,0,0]) ########1
        elif i_num==2:
            state[0] = torch.tensor([0.4327, -0.6417,0,0,  0.0982, -0.1987,0,0,  -0.9518, -0.7371,0,0,  0.9501, -0.0265,0,0]) ##########2
        elif i_num==3:
            state[0] = torch.tensor([-1.0065, -2.5180,0,0,  0.2307, -0.5298,0,0, -1.1549, 1.2102 ,0,0, -1.0345,  0.4486,0,0]) ##########3
        elif i_num==4:
            state[0] = torch.tensor([-2.5772,  2.7328,0,0, -1.0371,  1.9477,0,0, -2.9643, -1.9241,0,0, -2.3554, -0.7662,0,0]) ##########4
        time = torch.zeros(T,1).to(self.device)
        for t in range(T):
            # state_d[t] = torch.Tensor([sin(0.1*self.dt*t),cos(0.1*self.dt*t),0.1*cos(0.1*self.dt*t),-0.1*sin(0.1*self.dt*t),-0.1*0.1*sin(0.1*self.dt*t),-0.1*0.1*cos(0.1*self.dt*t)])
            state_d[t] = torch.Tensor([1,1,0,0,0,0])
            form_state[t]= torch.Tensor([[sin(0.1*self.dt*t+0*np.pi/2),cos(0.1*self.dt*t+0*np.pi/2),0.1*cos(0.1*self.dt*t+0*np.pi/2),-0.1*sin(0.1*self.dt*t+0*np.pi/2),-0.1*0.1*sin(0.1*self.dt*t+0*np.pi/2),-0.1*0.1*cos(0.1*self.dt*t+0*np.pi/2)],
                                    [sin(0.1*self.dt*t+1*np.pi/2),cos(0.1*self.dt*t+1*np.pi/2),0.1*cos(0.1*self.dt*t+1*np.pi/2),-0.1*sin(0.1*self.dt*t+1*np.pi/2),-0.1*0.1*sin(0.1*self.dt*t+1*np.pi/2),-0.1*0.1*cos(0.1*self.dt*t+1*np.pi/2)],
                                    [sin(0.1*self.dt*t+2*np.pi/2),cos(0.1*self.dt*t+2*np.pi/2),0.1*cos(0.1*self.dt*t+2*np.pi/2),-0.1*sin(0.1*self.dt*t+2*np.pi/2),-0.1*0.1*sin(0.1*self.dt*t+2*np.pi/2),-0.1*0.1*cos(0.1*self.dt*t+2*np.pi/2)],
                                    [sin(0.1*self.dt*t+3*np.pi/2),cos(0.1*self.dt*t+3*np.pi/2),0.1*cos(0.1*self.dt*t+3*np.pi/2),-0.1*sin(0.1*self.dt*t+3*np.pi/2),-0.1*0.1*sin(0.1*self.dt*t+3*np.pi/2),-0.1*0.1*cos(0.1*self.dt*t+3*np.pi/2)]])

            q_d = state_d[t,0:2]
            dq_d=state_d[t,2:4]
            ddq_d=state_d[t,4:6]
            
            e=torch.zeros((self.agents,self._coord_dim))
            e_dot=torch.zeros((self.agents,self._coord_dim))
            E=torch.zeros((self.agents,self._coord_dim))
            dphi_de=torch.zeros((self.agents,self._action_dim))
            ddphi_dde=torch.zeros((self.agents,self._action_dim,self._action_dim))
            phi=torch.zeros((self.agents,self._coord_dim))
            dphi_dt=torch.zeros((self.agents,self._coord_dim))
            for i in range(self.agents):
                state_i=state[t,self._state_dim*i:self._state_dim*(i+1)]
                q_i=state_i[:self._coord_dim]
                dq_i=state_i[self._coord_dim:]
                form_q_i=form_state[t,i,:self._coord_dim]
                form_dq_i=form_state[t,i,self._coord_dim:self._state_dim]
                form_ddq_i=form_state[t,i,self._state_dim:]

                for j in range(self.agents):
                    state_j=state[t,self._state_dim*j:self._state_dim*(j+1)]
                    q_j=state_j[:self._coord_dim]
                    dq_j=state_j[self._coord_dim:]
                    form_q_j=form_state[t,j,:self._coord_dim]
                    form_dq_j=form_state[t,j,self._coord_dim:self._state_dim]
                    form_ddq_j=form_state[t,j,self._state_dim:] 


                    e[i]=e[i]+self.L[i,j]*(q_i-form_q_i-(q_j-form_q_j))
                    e_dot[i]=e_dot[i]+self.L[i,j]*(dq_i-form_dq_i-(dq_j-form_dq_j))
                e[i]=e[i]+self.B[i]*(q_i-form_q_i-q_d)
                e_dot[i]=e_dot[i]+self.B[i]*(dq_i-form_dq_i-dq_d)

                z1 = e[i].requires_grad_() 
                dphi_de[i]=AGF.jacobian(lambda x: self.phi(x, i),z1,create_graph=True).squeeze()
                phi[i]=form_dq_i-dphi_de[i]
                E[i]=dq_i-phi[i]
                u_dpot_1 = AGF.hessian(lambda x: self.phi(x, i),z1,create_graph=True).squeeze()
                ddphi_dde[i] = u_dpot_1.clone().squeeze()
                dphi_dt[i]=(form_ddq_i-e_dot[i]@ ddphi_dde[i])
            
            for i in range(self.agents):
                state_i=state[t,self._state_dim*i:self._state_dim*(i+1)]
                q_i=state_i[:self._coord_dim]
                dq_i=state_i[self._coord_dim:]
                u_phi=torch.zeros(self._action_dim)
                for j in range(self.agents):
                    u_phi=u_phi+self.L[i,j]*(dphi_de[i]-dphi_de[j])
                u_phi=u_phi+self.B[i]*dphi_de[i]
                u_damp=self.damping_module_list[i](E[i])

                gc=self.gravity_compensate(state_i)
                C=self.Correlation(state_i)
                M=self.Gravity(state_i)
                control[t,self._action_dim*i:self._action_dim*(i+1)] = gc +(self.B[i]*ddq_d-e_dot[i]@ ddphi_dde[i]) @ M.t() + phi[i] @ C.t()-u_damp-u_phi 
                state[t+1,self._state_dim*i:self._state_dim*(i+1)] = self.f(state_i,control[t,self._action_dim*i:self._action_dim*(i+1)])

            time[t]=self.dt*t
            error.append(e)
        state_error=np.stack([tensor.detach().numpy() for tensor in error])
        e1=(state_error[-3000:,:,:]**2).sum(1).sum(1)
        # z1= state_error[-7000:,0]**2+state_error[-7000:,1]**2
        # z1_max = z1.max().detach().numpy() 
        print("stable tracking error:", e1.max())
        # np.save("dynamic_formation/directed_graph/notrbf_"+current_time+"_train_state_d.npy",state_d.clone().detach().numpy())
        # np.save("dynamic_formation/directed_graph/notrbf_"+current_time+"_train_time.npy",time.clone().detach().numpy())
        np.save("dynamic_formation/directed_graph/notrbf_"+current_time+"_train_S_"+str(a_print)+"_"+str(i_num)+".npy",state.clone().detach().numpy())
        np.save("dynamic_formation/directed_graph/notrbf_"+current_time+"_train_S_"+str(a_print)+"_"+str(i_num)+".npy",time.clone().detach().numpy())
        np.save("dynamic_formation/directed_graph/notrbf_"+current_time+"_train_S_"+str(a_print)+"_"+str(i_num)+".npy",form_state.clone().detach().numpy())
        # np.save("dynamic_formation/directed_graph/notrbf_"+current_time+"_train_state_n4_Lno13_24_"+str(self.rbf_module_list[0].num_centers)+'_'+str(i_num)+".npy",state.clone().detach().numpy())
        # np.save("dynamic_formation/directed_graph/notrbf_"+current_time+"_train_time_n4_Lno13_24_"+str(self.rbf_module_list[0].num_centers)+'_'+str(i_num)+".npy",time.clone().detach().numpy())

        # plt.figure()
        # fig,ax= plt.subplots(1,1,sharex=True)
        # ax.set_xlabel(r'$\beta_1$', fontsize=20)
        # ax.set_ylabel(r'$\beta_2$',fontsize=20)
        # ax.grid(linestyle='-')
        # ax.plot(state[:T,0].detach().numpy(),state[:T,1].detach().numpy(), color='blue',linewidth=1.5)#, label='$agent1$')
        # ax.plot(state[:T,4].detach().numpy(),state[:T,5].detach().numpy(), color='green',linewidth=1.5)#, label='$agent2$')
        # ax.plot(state[:T,8].detach().numpy(),state[:T,9].detach().numpy(), color='yellow',linewidth=1.5)#, label='$agent3$')
        # ax.plot(state[:T,12].detach().numpy(),state[:T,13].detach().numpy(), color='red',linewidth=1.5)#, label='$agent4$')
        # # ax[1].grid(linestyle='-')
        # # ax[1].plot(time[:T],state_error[:T,0].detach(), color='tab:blue',linewidth=1.5)#, label=r'$\beta_1$-$\beta_1^{d}$')
        # # ax[1].plot(time[:T],state_error[:T,1].detach(), color='tab:orange',linewidth=1.5)#, label=r'$\beta_2$-$\beta_2^{d}$')
        # ax.legend()

        plt.figure()
        fig,ax= plt.subplots(2,2,sharex=True)
        ax[1,1].set_xlabel('time',fontsize=20)
        ax[0,0].set_ylabel(r'$\beta_1$', fontsize=20)
        ax[0,1].set_ylabel(r'$\beta_2$',fontsize=20)
        ax[0,0].grid(linestyle='-')    

        # ax[0].plot(state[:T,0].detach().numpy(),state[:T,1].detach().numpy(), color='blue',linewidth=1.5)#, label='$agent1$')
        # ax[0].plot(state[:T,4].detach().numpy(),state[:T,5].detach().numpy(), color='green',linewidth=1.5)#, label='$agent2$')
        # ax[0].plot(state[:T,8].detach().numpy(),state[:T,9].detach().numpy(), color='yellow',linewidth=1.5)#, label='$agent3$')
        # ax[0].plot(state[:T,12].detach().numpy(),state[:T,13].detach().numpy(), color='red',linewidth=1.5)#, label='$agent4$')

        ax[0,0].plot(time[:T].detach().numpy(),state[:T,0].detach().numpy(), color='blue',linewidth=1.5)#, label='$agent1$')
        ax[0,0].plot(time[:T].detach().numpy(),state[:T,4].detach().numpy(), color='green',linewidth=1.5)#, label='$agent2$')
        ax[0,0].plot(time[:T].detach().numpy(),state[:T,8].detach().numpy(), color='yellow',linewidth=1.5)#, label='$agent3$')
        ax[0,0].plot(time[:T].detach().numpy(),state[:T,12].detach().numpy(), color='red',linewidth=1.5)#, label='$agent4$')
        ax[0,1].plot(time[:T].detach().numpy(),state[:T,1].detach().numpy(), color='blue',linewidth=1.5)#, label='$agent1$')
        ax[0,1].plot(time[:T].detach().numpy(),state[:T,5].detach().numpy(), color='green',linewidth=1.5)#, label='$agent2$')
        ax[0,1].plot(time[:T].detach().numpy(),state[:T,9].detach().numpy(), color='yellow',linewidth=1.5)#, label='$agent3$')
        ax[0,1].plot(time[:T].detach().numpy(),state[:T,13].detach().numpy(), color='red',linewidth=1.5)#, label='$agent4$')
        ax[1,0].plot(time[:T].detach().numpy(),state_d[:T,0].detach().numpy()+form_state[:T,0,0].detach().numpy()-state[:T,0].detach().numpy(), color='red',linewidth=1.5)#, label='$d-agent1$')
        ax[1,0].plot(time[:T].detach().numpy(),state_d[:T,0].detach().numpy()+form_state[:T,1,0].detach().numpy()-state[:T,4].detach().numpy(), color='blue',linewidth=1.5)#, label='$d-agent2$')
        ax[1,0].plot(time[:T].detach().numpy(),state_d[:T,0].detach().numpy()+form_state[:T,2,0].detach().numpy()-state[:T,8].detach().numpy(), color='green',linewidth=1.5)#, label='$d-agent3$')
        ax[1,0].plot(time[:T].detach().numpy(),state_d[:T,0].detach().numpy()+form_state[:T,3,0].detach().numpy()-state[:T,12].detach().numpy(), color='yellow',linewidth=1.5)#, label='$d-agent4$')
        # ax[0].plot(time[:T],state[:T,12], color='red',linewidth=1.5)#, label='$agent4$')
        ax[1,1].plot(time[:T].detach().numpy(),state_d[:T,1].detach().numpy()+form_state[:T,0,1].detach().numpy()-state[:T,1].detach().numpy(), color='red',linewidth=1.5)#, label='$d-agent1$')
        ax[1,1].plot(time[:T].detach().numpy(),state_d[:T,1].detach().numpy()+form_state[:T,1,1].detach().numpy()-state[:T,5].detach().numpy(), color='blue',linewidth=1.5)#, label='$d-agent2$')
        ax[1,1].plot(time[:T].detach().numpy(),state_d[:T,1].detach().numpy()+form_state[:T,2,1].detach().numpy()-state[:T,9].detach().numpy(), color='green',linewidth=1.5)#, label='$d-agent3$')
        ax[1,1].plot(time[:T].detach().numpy(),state_d[:T,1].detach().numpy()+form_state[:T,3,1].detach().numpy()-state[:T,13].detach().numpy(), color='yellow',linewidth=1.5)#, label='$d-agent4$')
        ax[1,1].grid(linestyle='-')
        ax[0,1].grid(linestyle='-')
        ax[1,0].grid(linestyle='-')
        # ax.legend()
        ax[0,1].legend()
        ax[1,1].legend()
        ax[0,0].legend()
        ax[1,0].legend()
        

        # ax[1].legend()
        plt.savefig("dynamic_formation/directed_graph/notrbf_"+current_time+"_train_S_"+str(a_print)+"_"+str(i_num)+".pdf", format='pdf',bbox_inches="tight")
        # plt.savefig("dynamic_formation/directed_graph/notrbf_"+current_time+"_train_n4_trajectories_'+str(self.rbf_module_list[0].num_centers)+'_'+str(i_num)+'.pdf', format='pdf',bbox_inches="tight")
        
        # plt.show()
        return state

if __name__ == '__main__':
        seed=3
        pl.seed_everything(seed)
        training_data = torch.Tensor([[0,0,0,0,1,1,0,0,-1,1,0,0,1,-1,0,0]])
        train_dataloader = DataLoader(training_data, batch_size=1)
        

        train_model=leaderfollower_NBS_tracking_control_learner()
        # train_model.to(device)
        # train_model.plottrajetory(1)
        train_model=leaderfollower_NBS_tracking_control_learner()
        # trainer = pl.Trainer(accelerator="cpu",
        #                         callbacks=[], max_epochs=200)
        # trainer.fit(train_model,train_dataloader)
        # trainer.save_checkpoint("dynamic_formation/directed_graph/notrbf_"+current_time+".ckpt")
        # train_model.plottrajetory(1)
        for i in range(1,5):
            seed=3
            pl.seed_everything(seed)
            train_model=leaderfollower_NBS_tracking_control_learner()
            train_model.plottrajetory(i)

        # torch.autograd.set_detect_anomaly(True)
        # alpha=np.linspace(0.05,2,10)
        # for a in alpha:
        #     pl.seed_everything(seed)
        #     train_model=leaderfollower_NBS_tracking_control_learner()
        #     trainer = pl.Trainer(accelerator="cpu",
        #                         callbacks=[], max_epochs=200)
        #     train_model.set_alpha(a)
        #     trainer.fit(train_model,train_dataloader)
        #     trainer.save_checkpoint("dynamic_formation/directed_graph/notrbf_"+current_time+"_S_"+str(a)+".ckpt")
        #     train_model.plottrajetory(1,a)

        # seed=3
        # pl.seed_everything(seed)
        # test_model=leaderfollower_NBS_tracking_control_learner.load_from_checkpoint("not_rbf/notrbf"+current_time+".ckpt")
        # test_model.plottrajetory(1)

        # state=np.load('leaderless_state_n4_0314.npy')
        # # T=state.shape[0]
        # plt.figure()
        # fig,ax= plt.subplots(1,1,sharex=True)
        # ax.set_xlabel(r'$\beta_1$', fontsize=20)
        # ax.set_ylabel(r'$\beta_2$',fontsize=20)


        # ax.grid(linestyle='-')
        
        # ax.plot(state[:,0],state[:,1], color='blue',linewidth=1.5)#, label='$agent1$')
        # ax.plot(state[:,4],state[:,5], color='green',linewidth=1.5)#, label='$agent2$')
        # ax.plot(state[:,8],state[:,9], color='yellow',linewidth=1.5)#, label='$agent3$')
        # ax.plot(state[:,12],state[:,13], color='red',linewidth=1.5)#, label='$agent4$')
        # ax[1].grid(linestyle='-')
        # ax[1].plot(time[:T],state_error[:T,0].detach(), color='tab:blue',linewidth=1.5)#, label=r'$\beta_1$-$\beta_1^{d}$')
        # ax[1].plot(time[:T],state_error[:T,1].detach(), color='tab:orange',linewidth=1.5)#, label=r'$\beta_2$-$\beta_2^{d}$')
        # ax.legend()
        # # ax[1].legend()
        # plt.savefig('leaderless_fig/leaderless_TLPRA_trajectories_1.pdf', format='pdf',bbox_inches="tight")

        # test_model.caculate_stable_error()

        ### different alpha in training contrast
        
        
        # alpha=np.linspace(0.05,2,40)
        
        # training_data = torch.Tensor([[0,0,0,0]])
        # train_dataloader = DataLoader(training_data, batch_size=1)
        
  
        # error_max=[]
        # trainers=locals()
        # models = locals()
        # for a in alpha:
        #     pl.seed_everything(seed)
        #     models['model_'+str(a)] = leaderfollower_NBS_tracking_control_learner()
        #     models['model_'+str(a)].set_alpha(a)
        #     # models['model_'+str(a)].caculate_stable_error()
        #     trainers['trainer_'+str(a)] = pl.Trainer(accelerator="cpu", num_nodes=1,
        #                     callbacks=[], max_epochs=200)
        #     trainers['trainer_'+str(a)].fit(models['model_'+str(a)], train_dataloader)
        #     trainers['trainer_'+str(a)].save_checkpoint("NBS_1_final_2link_1_alpha_0.1_4_40_"+str(a)+".ckpt")
        #     print(a,"trianing complete----\n")

        #     # models['model_'+str(a)].caculate_stable_error()
        #     # print("before test------\n")
        #     # model = leaderfollower_NBS_tracking_control_learner()
        #     # model.set_alpha(a)
        #     # trainer.fit(model, train_dataloader)
            
        #     train_model=leaderfollower_NBS_tracking_control_learner().load_from_checkpoint(
        #     checkpoint_path="NBS_1_final_2link_1_alpha_0.1_4_40_"+str(a)+".ckpt")
        #     error_max.append(train_model.caculate_stable_error())

        # error_alpha=np.array(error_max)
        # np.save('error_alpha.npy',error_alpha)

        # test_model=leaderfollower_NBS_tracking_control_learner().load_from_checkpoint(
        #     checkpoint_path="NBS_1_tracking_model_2link_1_alpha_u_0_4_11_01_4.0.ckpt")
        # test_model.plottrajetory()

        # error_alpha = np.load('error_alpha.npy')
        # bound = 0.5*2/(alpha**2)
        # z1=error_alpha
        # _, ax=plt.subplots(1,1, sharex=True)
        # ax.scatter(alpha, z1,color='tab:red')#, label=r'${\||z_1\||^2}$')
        # ax.plot(alpha, bound,color='tab:blue',linewidth=1.5)#, label='bound')
        # # ax.plot(alpha, error_alpha[:,0],color='tab:blue',linewidth=1.5)#, label=r'${\beta_1-\beta_1^{desire}}$')
        # # ax.plot(alpha, error_alpha[:,1],color='tab:orange',linewidth=1.5)#, label=r'${\beta_2-\beta_2^{desire}}$')
        # ax.set_ylabel('maximal steady-state tracking error',fontsize=20)
        # ax.set_xlabel(r'${\alpha}$',fontsize=20)
        # # ax.set_ylim(0,0.5)
        # ax.grid(linestyle='-')
        # ax.legend()

        # axins = inset_axes(ax, width='50%',height='30%',loc='center right')
        # axins.plot(alpha[1:], bound[1:],color='tab:blue')
        # axins.scatter(alpha[1:], z1[1:],color='tab:red')
        # axins.set_xlim(1.5,2)
        # axins.set_ylim(0,0.5)
        # axins.spines['top'].set_color('red')
        # axins.spines['right'].set_color('red')
        # axins.spines['bottom'].set_color('red')
        # axins.spines['left'].set_color('red')
        # axins.plot([8,10,10,4,4],[0,0,0.1,0.1,0], 'red')
        # # plt.show()
        # plt.savefig('alpha_error_1_0.05_2_40_1D_bound.pdf', format='pdf',bbox_inches="tight")

  