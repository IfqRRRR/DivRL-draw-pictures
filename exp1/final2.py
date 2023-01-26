import imageio
from tqdm import tqdm
import numpy as np
from matplotlib import cm
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
AGENTS = 3
EPISODES = 30
QUALITY = 500
COLORS = ['red','blue','green','yellow','brown']
np.random.seed(4)

# utils
A = np.array([
    [1.,1.,1.],
    [0.,1.,.5],
    [0.,0.,np.sqrt(3)/2]
])

inver_A = np.linalg.inv(A)

def p2x(p):
    assert p.shape[0]==3
    flatten_p = p.reshape([3,-1])
    flatten_x = A @ flatten_p
    x = flatten_x.reshape([3,*p.shape[1:]])
    return x[1:]

def x2p(x):
    assert x.shape[0]==2
    flatten_x = x.reshape([2,-1])
    one_vec = np.ones([1, flatten_x.shape[1]])
    flatten_x = np.concatenate([one_vec, flatten_x], 0)
    flatten_p = inver_A @ flatten_x
    p = flatten_p.reshape([3,*x.shape[1:]])
    return p

'''
plot training process
gt   : ground truth rewards
q    : optimal policy for soft-learning q(a|s_0)
v    : current q function
p_sz : current policy p(a|s_0,z)
ps   : current avg policy p(a|s_0)
'''
# train
plt.figure(figsize=[8.,2.])
gt = np.array([[1.,1.,-10] for _ in range(AGENTS)])
q = np.exp(gt) / np.exp(gt).sum(-1,keepdims=True)
v = np.random.rand(AGENTS, 3) 
p_sz = np.exp(v) / np.exp(v).sum(-1,keepdims=True)
counting = 0
plt.axis('off')
for episode in tqdm(range(EPISODES)):

    
    
    # train step 
    p_sz = np.exp(v) / np.exp(v).sum(-1,keepdims=True)
    p_s = np.mean(p_sz, 0)
    v += (p_sz - p_s) *.2         # max D_KL(p(a|s,z)||p(a|s))
    # v -= (p_s - q)             # min D_KL(p||q)

    if episode not in [1, 13, 26]:
      continue
    # plot D_KL(p(a|s_0,z=0)||p(a|s_0))
    img_x = np.arange(QUALITY) / (QUALITY-1)
    img_y = np.arange(QUALITY) / (QUALITY-1) * np.sqrt(3) / 2
    img_x, img_y = np.meshgrid(img_x, img_y)
    img_coor = np.stack([img_x,img_y], 0) 
    img_prob = x2p(img_coor) 
    img_ps = [img_prob]
    for a_id in range(1,AGENTS):
        img_psz = p_sz[a_id].reshape([3,1,1])
        img_psz = img_psz.repeat(QUALITY,1).repeat(QUALITY,2)
        img_ps.append(img_psz)
    img_ps = np.stack(img_ps, 0).mean(0) 
    img_ps = np.zeros([3, QUALITY, QUALITY])
    for i in range(3):
      img_ps[i] = p_s[i]
    kl_dis = np.sum(img_prob*np.log(img_prob) - img_prob*np.log(img_ps), 0)
    contour = np.where(img_prob[2]>0, kl_dis, np.nan)
    plt.contourf(img_x + counting * 1.3, img_y, contour, 10, cmap='RdGy')

    # plot D_KL(p(.|s_0)||q(.|s_0))
    # img_x = np.arange(QUALITY) / (QUALITY-1)
    # img_y = np.arange(QUALITY) / (QUALITY-1) * np.sqrt(3) / 2
    # img_x, img_y = np.meshgrid(img_x, img_y)
    # img_coor = np.stack([img_x,img_y], 0) 
    # img_prob = x2p(img_coor) 
    # img_ps = q[0].reshape([3,1,1])
    # img_ps = img_ps.repeat(QUALITY,1).repeat(QUALITY,2)
    # kl_dis = np.sum(img_prob*np.log(img_prob) - img_prob*np.log(img_ps), 0)
    # contour = np.where(img_prob[2]>0, kl_dis, np.nan)
    # plt.contourf(img_x, img_y, contour, 20, cmap='RdGy')

    # plot triangle and points
    img_x = p2x(p_sz.T) # shape=(2, AGENTS)
    for a_id in range(AGENTS):
        # plt.scatter(img_x[0,a_id], img_x[1,a_id], c=COLORS[a_id], s=5.)
        plt.scatter(img_x[0,a_id] + counting * 1.3, img_x[1,a_id], c=COLORS[a_id], s=5.)
    # plt.scatter(p2x(p_s)[0], p2x(p_s)[1], c='black')
    plt.scatter(p2x(p_s)[0] + counting * 1.3, p2x(p_s)[1], c='black')
    # triangles = tri.Triangulation(A[1], A[2])
    # plt.triplot(triangles,'-')
    triangles = tri.Triangulation(A[1] + counting * 1.3, A[2])
    plt.triplot(triangles,'-')
    counting += 1
    
plt.savefig('final2.png',bbox_inches='tight')
plt.savefig('final2.pdf',bbox_inches='tight')

plt.close()
# images = []
# for episode in range(EPISODES):
#     images.append(imageio.imread('./img/{:06d}.png'.format(episode)))
# imageio.mimsave('result.gif', images, fps=10)



'''
plot 3d kl-distance
'''
# A = np.array([[0., 1., 1/2],
#               [0., 0., np.sqrt(3)/2]])

# def p_to_xy(p):
#     assert p.shape[0] == 3
#     if len(p.shape) > 2:
#         flatten_p = p.reshape([p.shape[0], -1])
#         result = A @ flatten_p
#         return result.reshape([-1, *p.shape[1:]])
#     return A @ p

# gt = np.array([1.,1.,-1e6])
# q = np.exp(gt) / np.exp(gt).sum(-1) + 1e-6
# q = q.reshape([-1,1,1])
# x = np.arange(100) / 99.
# y = np.arange(100) / 99.
# x, y = np.meshgrid(x, y)
# p = np.stack([x,y,1-x-y], 0) + 1e-6
# kl_dis = np.sum(p*np.log(p) - p*np.log(q), 0)
# kl_dis[np.where(1-x-y<-1e-6)] = np.nan
# xy = p_to_xy(p)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(xy[0,:], xy[1,:], kl_dis)
# ax.contour(xy[0,:], xy[1,:], kl_dis, offset=0.)
# ax.scatter(p_to_xy(q)[0,0,0], p_to_xy(q)[1,0,0], 0.)
# plt.show()
