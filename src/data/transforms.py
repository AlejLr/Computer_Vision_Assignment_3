from torchvision import transforms

def get_train_transforms(input_size):
    '''
    Very basic train transforms for the baseline
    Is center frame, so we rely mostly on spatial augmentation
    '''
    
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
def get_val_transforms(input_size):
    '''
    Very basic val transforms for the baseline
    Is center frame, so we rely mostly on spatial augmentation
    '''
    
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])