from typing import Any
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from numpy.linalg import inv
import imageio
# import mediapy as media
# from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.autograd.functional as AGF

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
import time

import torch.optim as optim

from torch.autograd.functional import jacobian, hessian
from torchdiffeq import odeint as tor_odeint
from torchdiffeq import odeint_adjoint as tor_odeintadj
from functools import partial

from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import EarlyStopping
import math
import torch.linalg as linalg
# from scipy.integrate import odeint
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
from numpy import cos, sin, arccos, arctan2, sqrt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mujoco_base import MuJoCoBase

from datetime import datetime 

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size":20,
    "font.sans-serif": ["Helvetica"]})

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

target_pos=np.array([1.04471173e-14, -1.12814687e+01, -2.14543790e-14, -1.21697198e+01 ,3.96139652e-15,  7.16583248e+00,  3.60969650e-30])
# target_pos=np.array([1.99310014e-15, -5.68989112e+00, -2.21674702e-15,  7.81954841e+00,1.12400095e-15,  7.03369792e+00, -2.05696290e-30])

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
        hidden_num=32
        input_num=7
        self.w_z0=nn.Parameter(torch.Tensor(input_num,hidden_num).uniform_(-0.1,0.1))
        self.w_y1=nn.Parameter(torch.Tensor(input_num,hidden_num).uniform_(-0.1,0.1))
        self.w_y2=nn.Parameter(torch.Tensor(input_num,hidden_num).uniform_(-0.1,0.1))
        self.w_yout=nn.Parameter(torch.Tensor(input_num,1).uniform_(-0.1,0.1))
        self.w_z1=nn.Parameter(torch.Tensor(hidden_num,hidden_num).uniform_(0,0.1))
        self.w_z2=nn.Parameter(torch.Tensor(hidden_num,hidden_num).uniform_(0,0.1))
        self.w_zout=nn.Parameter(torch.Tensor(hidden_num,1).uniform_(0,0.1))
        self.srleu=Srelu()
        
    def forward(self, z):
        
        z0 = z.clone()
        z1 = z0 @ self.w_z0 
        z1s =self.srleu(z1)
        z2 = z1s @ torch.relu(self.w_z1)   +z0 @ self.w_y1 
        z2s = self.srleu(z2)
        z3 =  z2s @ torch.relu(self.w_z2)  + z0 @ self.w_y2
        z3s = self.srleu(z3)
        zout =  z3s @ torch.relu(self.w_zout)  + z0 @ self.w_yout
        zouts = self.srleu(zout)
        # z4 = self.w_z3 * z0 +self.b_z3 +self.w_y3 * z3s +self.b_y3
        # z4s = self.srleu(z4)
            #z = torch.tanh(self._y_layers[i+1](z0) + self._z_layers[i](z))
        return zouts
 
class Damping(nn.Module):  # represents the controller gain
    def __init__(self):
        super(Damping, self).__init__()
        N = 7
        self.offdiag_output_dim = N*(N-1)//2
        self.diag_output_dim = N
        self.output_dim = self.offdiag_output_dim + self.diag_output_dim
        damp_min=torch.ones(N)*0.5
        self.damp_min = damp_min
        hidden_num=64
        # w_d1=torch.tensor(2,32).uniform_(-0.01,0.01)

        self.w_d1=nn.Parameter(torch.Tensor(N,hidden_num).uniform_(-0.1,0.1))
        self.w_d2=nn.Parameter(torch.Tensor(hidden_num,hidden_num).uniform_(-0.1,0.1))
        self.w_d3=nn.Parameter(torch.Tensor(hidden_num,self.diag_output_dim).uniform_(-0.1,0.1))
        self.w_o1=nn.Parameter(torch.Tensor(N,hidden_num).uniform_(-0.1,0.1))
        self.w_o2=nn.Parameter(torch.Tensor(hidden_num,hidden_num).uniform_(-0.1,0.1))
        self.w_o3=nn.Parameter(torch.Tensor(hidden_num,self.offdiag_output_dim).uniform_(-0.1,0.1))
        self.b_d1=nn.Parameter(torch.zeros(hidden_num))
        self.b_d2=nn.Parameter(torch.zeros(hidden_num))
        self.b_d3=nn.Parameter(torch.zeros(self.diag_output_dim))
        self.b_o1=nn.Parameter(torch.zeros(hidden_num))
        self.b_o2=nn.Parameter(torch.zeros(hidden_num))
        self.b_o3=nn.Parameter(torch.zeros(self.offdiag_output_dim))

    def forward(self, input):
        # todo this method is not ready for batch input data
        # x = input.view(1,-1)
        x = input
        x0=x.clone()
        z=x.clone()

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

class RBFNN(nn.Module):
    def __init__(self) -> None:
        super(RBFNN,self).__init__()
        self.coord_dim=7
        self.input_dim = input_dim=4*self.coord_dim
        self.hidden_dim = hidden_dim=32
        self.output_dim = output_dim=self.coord_dim
        self.num_centers = num_centers=2048
        self.dt=dt=0.01
        
        # RBF层的参数
        self.centers = torch.Tensor(num_centers, input_dim).normal_(-3,3)
        self.widths = torch.ones(num_centers)*3
        
        # 全连接层
        self.w = torch.randn(num_centers,output_dim)
        self.mu=torch.tensor([1])
        

    def forward(self, q,dq,e,E):
        # 计算RBF层的输出
        x=torch.cat((q.clone(),dq.clone(),e.clone(),E.clone())).unsqueeze(0)
        rbf = torch.exp(-torch.sum((x.unsqueeze(1) - self.centers)**2, dim=2) / (2*self.widths**2))
        
        # 计算全连接层的输出
        output = rbf@self.w
        return output
    
    def update_RBFNN(self,q,dq,phi,dphi,E):
        x=torch.cat((q.clone(),dq.clone(),phi.clone(),dphi.clone())).unsqueeze(0)
        rbf = torch.exp(-torch.sum((x.unsqueeze(1) - self.centers)**2, dim=2) / (2*self.widths**2))
        dotW=-100*(rbf.t()@E.clone().unsqueeze(0)+self.mu*self.w)
        dotmu=-0.5*self.mu
        self.w=self.w+self.dt*dotW
        self.mu=self.mu+self.dt*dotmu
  

class ManipulatorDrawing(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.simend = 70.0
        self.L=torch.tensor([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
        self.B=torch.tensor([1,0,0,0])
        # self.B_d=torch.diag(self.B)
        # self.form_state = torch.tensor([[0.5,0.5,0.5],[-0.5,-0.5,-0.5],[1,1,1],[-1,-1,-1]])
        self.agents=4
        self.icnn_module=ICNN()
        self.damping_module=Damping()
        self.rbf_module_list=[]
        for i in range(self.agents):
            self.rbf_module_list.append(RBFNN())

        coord_dim=7
        self._coord_dim = coord_dim
        self._state_dim = coord_dim*2
        self._action_dim = coord_dim
        self.init_quad_pot = 1.0,
        self.min_quad_pot = 1e-3,
        self.max_quad_pot = 1e1,
        self.S = torch.eye(coord_dim)*torch.Tensor([1.0])

        # self.icnn_module = ICNN(Controller_w)
        # param_normal=torch.load('new_model.pth')
        # self.damping_module = Damping(Controller_w)

        init_quad_pot_param = torch.ones(coord_dim)*torch.Tensor([1.0])
        self._quad_pot_param = init_quad_pot_param
        self._min_quad_pot_param = torch.Tensor([1e-3])
        self._max_quad_pot_param = torch.Tensor([1e1])

        # self.lnn=LNN()
        self.icnn_module=ICNN()
        self.damping_module=Damping()

        # self.model.opt.timestep= 0.001
        # self.dt = self.model.opt.timestep
        # self.m1 = self.model.body_mass[1]
        # self.m2 = self.model.body_mass[2]
        # self.l1 = 2*self.model.geom(1).size[1]
        # self.l2 = 2*self.model.geom(2).size[1]
        # self.g  = self.model.opt.gravity[2]


    def reset(self):
        # Set initial angle of pendulum
        self.data.qpos=np.zeros(self.model.nq)
        # self.data.qpos[0] = 0.0
        # self.data.qpos[1] = 0.0
        # self.data.qpos[2] = 0.0
        # self.data.qpos[3] = 0.0
        # self.data.qpos[4] = 0.0
        # self.data.qpos[5] = 0.0
        # self.data.qpos[6] = 0.0
        # self.data.qpos[7] = 0.0
        # self.data.qpos[8] = 0.0
        # self.data.qpos[9] = 0.0
        # self.data.qpos[10] = 0.0
        # self.data.qpos[11] = 0.0
        self.state= []
        self.state_error = []
        self.sim_time= []
        self.error=[]
        self.form_state = []
        
        
        # self.renderer.width=800
        # Set camera configuration
        self.cam.azimuth = 90
        self.cam.elevation = -15
        self.cam.distance = 8.0
        self.cam.lookat = np.array([0.0, 0.0, 3])

        mj.mj_forward(self.model, self.data)


        # mj.set_mjcb_control(self.controller)
    def phi(self, x):
        y=self.icnn_module(x)+ (x @ self.S @ x.t())

        return y
    

    def controller(self, model, data):
        """
        This function implements a P controller for tracking
        the reference motion.
        """
        state=torch.tensor(self.data.sensordata).float()
        # q=state[:3]
        # dq=state[3:6]
        self.sim_time.append(self.data.time)
        
        self.state.append(torch.cat([state[0:7],state[14:14+7],state[28:28+7],state[42:42+7]]).numpy())

        form_state=torch.zeros((self.agents,3*7))
        q_d = torch.Tensor(target_pos)
        e=torch.zeros((self.agents,self._coord_dim))
        e_dot=torch.zeros((self.agents,self._coord_dim))
        E=torch.zeros((self.agents,self._coord_dim))
        dphi_de=torch.zeros((self.agents,self._action_dim))
        ddphi_dde=torch.zeros((self.agents,self._action_dim,self._action_dim))
        phi=torch.zeros((self.agents,self._coord_dim))
        dphi_dt=torch.zeros((self.agents,self._coord_dim))
        control=torch.zeros((self.agents,self._coord_dim))

        for i in range(self.agents):
            state_i=state[self._state_dim*i:self._state_dim*(i+1)]
            q_i=state_i[:self._coord_dim]
            dq_i=state_i[self._coord_dim:]
            form_q_i=form_state[i,:self._coord_dim]
            form_dq_i=form_state[i,self._coord_dim:self._state_dim]
            form_ddq_i=form_state[i,self._state_dim:]

            for j in range(self.agents):
                state_j=state[self._state_dim*j:self._state_dim*(j+1)]
                q_j=state_j[:self._coord_dim]
                dq_j=state_j[self._coord_dim:]
                form_q_j=form_state[j,:self._coord_dim]
                form_dq_j=form_state[j,self._coord_dim:self._state_dim]
                form_ddq_j=form_state[j,self._state_dim:]
                e[i]=e[i]+self.L[i,j]*(q_i-form_q_i-(q_j-form_q_j))
                e_dot[i]=e_dot[i]+self.L[i,j]*(dq_i-form_dq_i-(dq_j-form_dq_j))
            e[i]=e[i]+self.B[i]*(q_i-form_q_i-q_d)
            e_dot[i]=e_dot[i]+self.B[i]*(dq_i-form_dq_i)

            z1 = e[i].requires_grad_() 
            # psi = self.phi(z1)
            # u_pot = torch.autograd.grad(psi,z1,create_graph=True,retain_graph=True)
            # dphi_de[i]= u_pot[0]
            dphi_de[i]=AGF.jacobian(self.phi,z1,create_graph=True).squeeze()
            phi[i]=form_dq_i-dphi_de[i]
            E[i]=dq_i-phi[i]
            ddphi_dde[i] = AGF.hessian(self.phi,z1,create_graph=True).squeeze()
            dphi_dt[i]=(form_ddq_i-e_dot[i]@ ddphi_dde[i])
        
        for i in range(self.agents):
            state_i=state[self._state_dim*i:self._state_dim*(i+1)]
            q_i=state_i[:self._coord_dim]
            dq_i=state_i[self._coord_dim:]
            u_phi=torch.zeros(self._action_dim)
            for j in range(self.agents):
                u_phi=u_phi+self.L[i,j]*(dphi_de[i]-dphi_de[j])
            u_phi=u_phi+self.B[i]*dphi_de[i]
            u_damp=self.damping_module(E[i])
            u_rbf=self.rbf_module_list[i](q_i,dq_i,phi[i],dphi_dt[i])
            control[i] = u_rbf-u_damp-u_phi
            

            self.rbf_module_list[i].update_RBFNN(q_i,dq_i,phi[i],dphi_dt[i],E[i])

        data.ctrl[0*7+0] = control[0,0].detach().numpy()
        data.ctrl[0*7+1] = control[0,1].detach().numpy()
        data.ctrl[0*7+2] = control[0,2].detach().numpy()
        data.ctrl[0*7+3] = control[0,3].detach().numpy()
        data.ctrl[0*7+4] = control[0,4].detach().numpy()
        data.ctrl[0*7+5] = control[0,5].detach().numpy()
        data.ctrl[0*7+6] = control[0,6].detach().numpy()
        data.ctrl[1*7+0] = control[1,0].detach().numpy()
        data.ctrl[1*7+1] = control[1,1].detach().numpy()
        data.ctrl[1*7+2] = control[1,2].detach().numpy()
        data.ctrl[1*7+3] = control[1,3].detach().numpy()
        data.ctrl[1*7+4] = control[1,4].detach().numpy()
        data.ctrl[1*7+5] = control[1,5].detach().numpy()
        data.ctrl[1*7+6] = control[1,6].detach().numpy()
        data.ctrl[2*7+0] = control[2,0].detach().numpy()
        data.ctrl[2*7+1] = control[2,1].detach().numpy()
        data.ctrl[2*7+2] = control[2,2].detach().numpy()
        data.ctrl[2*7+3] = control[2,3].detach().numpy()
        data.ctrl[2*7+4] = control[2,4].detach().numpy()
        data.ctrl[2*7+5] = control[2,5].detach().numpy()
        data.ctrl[2*7+6] = control[2,6].detach().numpy()
        data.ctrl[3*7+0] = control[3,0].detach().numpy()
        data.ctrl[3*7+1] = control[3,1].detach().numpy()
        data.ctrl[3*7+2] = control[3,2].detach().numpy()
        data.ctrl[3*7+3] = control[3,3].detach().numpy()
        data.ctrl[3*7+4] = control[3,4].detach().numpy()
        data.ctrl[3*7+5] = control[3,5].detach().numpy()
        data.ctrl[3*7+6] = control[3,6].detach().numpy()



    def gif(self):
        # frames=[]
        while 1:
            simstart = self.data.time

            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                self.controller(self.model,self.data)
                mj.mj_step(self.model, self.data)
                # self.renderer.update_scene(self.data,self.cam)
            print(self.data.time,self.data.xpos[9],self.data.xpos[-1])
            if self.data.time >= self.simend:
                break
            # width=800
            # height=800
            # fig=self.renderer.render()
            # frames.append(fig.copy())
            
        # imageio.mimsave("simulation_1.gif", frames, duration=1.0/60)

    def simulate(self):
        frames=[]
        while not glfw.window_should_close(self.window):
            simstart = self.data.time
            

            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                
                mj.mj_step(self.model, self.data)
            self.controller(self.model,self.data)

            if self.data.time >= self.simend:
                break

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            # fig = mj.MjvFigure()
            # mj.mjv_defaultFigure(fig)


            # Update scene and render
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            # mj.mjr_figure(viewport, fig, self.context)
            # frames.append(fig)
            # media.show_image(viewport)
            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()


        # imageio.imwrite("sample.gif", frames)

    def plot(self):
        _, ax=plt.subplots(7,1, sharex=True)
        sim_time=np.array(self.sim_time)
        state = np.array(self.state)
        # state_error=np.array(self.error)
        # state_d = np.array(self.state_d)
        # state_error=np.array(self.state_error)1

        np.save("mujoco_"+current_time+"_state_02.npy",state)
        np.save("sim_"+current_time+"_time_02",sim_time)


        for i in range(7):
            ax[i].plot(sim_time, state[:,i+7*0],color='tab:blue',linewidth=1.5, label=str(i)+r'${\beta}$')
            ax[i].plot(sim_time, state[:,i+7*1],color='tab:orange',linewidth=1.5, label=str(i)+r'${\beta_2}$')
            ax[i].plot(sim_time, state[:,i+7*2],color='tab:green',linewidth=1.5, label=str(i)+r'${\beta_3}$')
            ax[i].plot(sim_time, state[:,i+7*3],color='tab:yellow',linewidth=1.5, label=str(i)+r'${\beta_4}$')


            ax[i].grid(linestyle='-')
            ax[i].legend()

        # for i in range(self.agents):
        #     ax[i].plot(sim_time, state[:,0+7*i],color='tab:blue',linewidth=1.5, label=r'${\beta_1}$')
        #     ax[i].plot(sim_time, state[:,1+7*i],color='tab:orange',linewidth=1.5, label=r'${\beta_2}$')
        #     ax[i].plot(sim_time, state[:,2+7*i],color='tab:green',linewidth=1.5, label=r'${\beta_3}$')
        #     ax[i].plot(sim_time, state[:,3+7*i],color='tab:yellow',linewidth=1.5, label=r'${\beta_4}$')
        #     ax[i].plot(sim_time, state[:,4+7*i],color='tab:red',linewidth=1.5, label=r'${\beta_5}$')
        #     ax[i].plot(sim_time, state[:,5+7*i],color='tab:pink',linewidth=1.5, label=r'${\beta_6}$')
        #     ax[i].plot(sim_time, state[:,6+7*i],color='tab:black',linewidth=1.5, label=r'${\beta_7}$')

        #     ax[i].grid(linestyle='-')
        #     ax[i].legend()


        # ax[1].plot(sim_time, state_error[:,0],color='tab:blue',linewidth=1.5, label=r'${\beta_1}-{\beta_1^{d}}$')
        # ax[1].plot(sim_time, state_error[:,1],color='tab:orange',linewidth=1.5, label=r'${\beta_2}-{\beta_2^{d}}$')
        # ax[1].plot(sim_time, state_error[:,2],color='tab:green',linewidth=1.5, label=r'${\beta_3}-{\beta_3^{d}}$')
        # ax[1].set_ylabel('tracking errors',fontsize=20)
        
        
        ax[3].set_xlabel('time ',fontsize=20)

        # ax[0].set_xlim(0,150)
        # ax[1].set_xlim(0,150)

        
        # ax[1].grid(linestyle='-')
        # # ax[2].plot(sim_time, state_d[:,0],color='tab:blue',linewidth=1.5, label=r'${\beta_1}$')
        # # ax[2].plot(sim_time, state_d[:,1],color='tab:orange',linewidth=1.5, label=r'${\beta_2}$')
        # # ax[2].set_ylabel(r'idael robot link angles')
        # # ax[2].grid(linestyle='-')

        # ax[0].legend()
        # ax[1].legend()
        plt.savefig('dynamic_formation/mujoco_multiple_franka/'+current_time+'mujoco_4threelink.pdf',format='pdf')
        # plt.show()


def main():
    xml_path = "dynamic_formation/mujoco_multiple_threelink/four_franka.xml"
    sim = ManipulatorDrawing(xml_path)
    seed=3
    # sim.reset()
    # train_data=sim.get_dataset()
    # np.save("lnn_traindata.npy",train_data.numpy())

    # train_data=np.load("lnn_traindata.npy")
    # train_data=torch.tensor(train_data)
    # train_dataloader = DataLoader(train_data[:10000], batch_size=10)
    
    # pl.seed_everything(seed)
    # lnn_model=lnn_leaner()
    # trainer = pl.Trainer(accelerator="cpu", num_nodes=1,
    #                         callbacks=[], max_epochs=200)
    # trainer.fit(lnn_model, train_dataloader)
    # trainer.save_checkpoint("mujoco_picnn_lnn_threelink_02.ckpt")
    #00是200epochs，16neurons,无参数初始化。
    pl.seed_everything(seed)
    sim.reset()
    # sim.set_NN(LNN_model=test_lnn_model.lnn,icnn_model=test_NBS_model.icnn_module,damping_model=test_NBS_model.damping_module)
    sim.gif()
    # sim.simulate()
    sim.plot()

if __name__ == "__main__":
    main()
