class Summary(object):
    def summary(self, model="sdid"):
        if model == "sdid":
            att = self.hat_tau(model="sdid")
            print("------------------------------------------------------------")
            print("Syntetic Difference in Differences")
            print("")
            if self.sdid_se != None:
                print(f"point estimate: {att:.3f}  ({self.sdid_se:.3f})")
                print(
                    f"95% CI ({att - 1.96*self.sdid_se :.3f}, {att + 1.96*self.sdid_se:.3f})"
                )
            else:
                print(f"point estimate: {att:.3f}")
                return att
        elif model == "sc":
            att = self.hat_tau(model="sc")
            print("------------------------------------------------------------")
            print("Syntetic Control Method")
            print("")
            if self.sdid_se != None:
                print(f"point estimate: {att:.3f}  ({self.sc_se:.3f})")
                print(
                    f"95% CI ({att - 1.96*self.sc_se :.3f}, {att + 1.96*self.sc_se:.3f})"
                )
            else:
                print(f"point estimate: {att:.3f}")
            return att

        elif model == "did":
            att = self.hat_tau(model="did")
            print("------------------------------------------------------------")
            print("Difference in Differences")
            print("")
            if self.sdid_se != None:
                print(f"point estimate: {att:.3f}  ({self.did_se:.3f})")
                print(
                    f"95% CI ({att - 1.96*self.did_se :.3f}, {att + 1.96*self.did_se:.3f})"
                )
            else:
                print(f"point estimate: {att:.3f}")
                return att
            
        elif model == 'cleansc':
            att = self.hat_tau(model="sc")
            if self.sdid_se != None:
                return att, self.sc_se
            else:
                return att
        elif model == 'cleansdid':
            att = self.hat_tau(model="did")
            if self.sdid_se != None:
                return att, self.did_se
            else:
                return att
        elif model == 'cleansdid':
            att = self.hat_tau(model="sdid")
            if self.sdid_se != None:
                return att, self.sdid_se
            else:
                return att
            
