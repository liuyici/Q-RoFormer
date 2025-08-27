            for domain_idx in range(14):
                features_source = tok[domain_idx*args.batch_size:(domain_idx+1)*args.batch_size]
                features_s_Adver = Adver_network.ReverseLayerF.apply(features_source, args.gamma)#用这个替代features_source经过了反转层
                outputs_source_domain_D = self.domain_Discriminator(features_s_Adver)
                pred_src_domain_D.append(outputs_source_domain_D)
             
                # coral_loss = utils.CORAL_loss(features_source, features_target)
                mmd_b_loss += utils.marginal(features_source,tok_target)
                mmd_t_loss += utils.conditional(
                    features_source,
                    tok_target,
                    y_src[domain_idx].reshape((args.batch_size, 1)),
                    torch.nn.functional.softmax(outputs_target,dim = 1),
                    2.0,
                    5,
                    None)
            # 将每个源域的标签拼接起来
            pred_source_domain_D = torch.cat(pred_src_domain_D, dim=0)
            Domain_labels_source = torch.cat(Dy_src, dim=0)
            Adver_domain_labels_loss = criterion(pred_source_domain_D, Domain_labels_source.flatten())
            MMD_loss = 0.5*mmd_b_loss + 0.5*mmd_t_loss
