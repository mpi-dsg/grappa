#        file_d = open(f"/workspace/dgl_dsg/experiments/tools/sample_log/random_{args.graph_name}_{args.num_layers}_rank_{rank}_epoch_{epoch}_real_no_shuffle.txt", "w")

#                    if args.num_layers == 3:
#                        src, dst = blocks[2].edges()
#                        src = blocks[2].srcdata[dgl.NID][src]
#                        dst = blocks[2].dstdata[dgl.NID][dst]
#                        file_d.write(str(dst))
#                        file_d.write('\n')
#                        file_d.write(str(src))
#                        file_d.write('\n')
#                        file_d.write('-'*20)
#                        file_d.write('\n')

#                    src, dst = blocks[1].edges()
#                    src = blocks[1].srcdata[dgl.NID][src]
#                    dst = blocks[1].dstdata[dgl.NID][dst]
#                    file_d.write(str(dst))
#                    file_d.write('\n')
#                    file_d.write(str(src))
#                    file_d.write('\n')
#                    file_d.write('-'*20)
#                    file_d.write('\n')
#                    src, dst = blocks[0].edges()
#                    src = blocks[0].srcdata[dgl.NID][src]
#                    dst = blocks[0].dstdata[dgl.NID][dst]
#                    file_d.write(str(dst))
#                    file_d.write('\n')
#                    file_d.write(str(src))
#                    file_d.write('\n')
#                    file_d.write('-'*20)
#                    file_d.write('\n')
#                    file_d.write('-'*20)
#                    file_d.write('\n')

#        file_d.close()
